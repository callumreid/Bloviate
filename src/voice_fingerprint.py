"""
Voice fingerprinting module for Bloviate.
Uses speaker embeddings to verify that audio matches the enrolled user's voice.
"""

import inspect
import numpy as np
import torch
import torchaudio

# Apply compatibility patch for newer torchaudio versions
import torchaudio_patch


def _patch_huggingface_hub():
    """
    SpeechBrain 1.0 expects hf_hub_download(use_auth_token=...).
    Newer huggingface_hub removed that arg in favor of token=...
    Patch before importing SpeechBrain so its modules pick up the wrapper.
    """
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download as _hf_hub_download

        sig = inspect.signature(_hf_hub_download)
        supports_token = "token" in sig.parameters
        supports_use_auth_token = "use_auth_token" in sig.parameters

        def _wrapped_hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
            if token is None and use_auth_token is not None:
                token = use_auth_token
            try:
                if supports_token:
                    return _hf_hub_download(*args, token=token, **kwargs)
                if supports_use_auth_token:
                    return _hf_hub_download(*args, use_auth_token=use_auth_token, **kwargs)
                return _hf_hub_download(*args, **kwargs)
            except Exception as exc:
                filename = kwargs.get("filename")
                msg = str(exc)
                if filename == "custom.py" and ("404" in msg or "Entry Not Found" in msg):
                    raise ValueError("File not found on HF hub") from exc
                raise

        huggingface_hub.hf_hub_download = _wrapped_hf_hub_download
    except Exception:
        pass


_patch_huggingface_hub()

try:
    from speechbrain.inference import EncoderClassifier
except Exception:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except Exception:
        EncoderClassifier = None
from typing import Optional, List
import os
import pickle
from pathlib import Path
import shutil

from app_paths import (
    config_base_dir,
    legacy_repo_voice_profile_path,
    models_dir as default_models_dir,
    resolve_path,
)


class VoiceFingerprint:
    """
    Speaker verification using voice embeddings.
    Only accepts audio matching the enrolled voice profile.
    """

    def __init__(self, config: dict, model_dir: Optional[str] = None):
        self.config = config
        self.enabled = config['voice_fingerprint']['enabled']
        self.threshold = config['voice_fingerprint']['threshold']
        self.sample_rate = config['audio']['sample_rate']
        self.model_name = config['voice_fingerprint']['embedding_model']
        self.min_enrollment_samples = config['voice_fingerprint']['min_enrollment_samples']

        configured_model_dir = (
            model_dir
            or config.get("voice_fingerprint", {}).get("model_dir")
            or os.getenv("BLOVIATE_MODEL_DIR")
        )
        if configured_model_dir:
            self.model_dir = resolve_path(
                str(configured_model_dir),
                base_dir=config_base_dir(config),
            )
        else:
            self.model_dir = default_models_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.profile_path = self.model_dir / "voice_profile.pkl"
        self.legacy_profile_path = legacy_repo_voice_profile_path()

        # Load speaker embedding model
        print("Loading speaker embedding model...")
        try:
            if EncoderClassifier is None:
                raise RuntimeError("SpeechBrain EncoderClassifier unavailable")
            self.encoder = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=str(self.model_dir / "pretrained")
            )
            print("Speaker embedding model loaded")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.enabled = False
            self.encoder = None

        # Load existing voice profile if available
        self.enrolled_embeddings: List[np.ndarray] = []
        self.reference_embedding: Optional[np.ndarray] = None
        self.load_profile()

    def extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio signal as numpy array

        Returns:
            Embedding vector or None if extraction fails
        """
        if not self.enabled or self.encoder is None:
            return None

        try:
            # Ensure audio is 1D
            if len(audio.shape) > 1:
                audio = audio.squeeze()

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Add batch dimension
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Ensure embeddings are 1D
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2

        return float(similarity)

    def verify_speaker(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Verify if audio matches the enrolled voice.

        Args:
            audio: Audio signal to verify

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not self.enabled or self.reference_embedding is None:
            return True, 1.0  # Pass-through if not enabled or not enrolled

        # Extract embedding from input audio
        embedding = self.extract_embedding(audio)
        if embedding is None:
            return False, 0.0

        # Compare with reference embedding
        similarity = self.compute_similarity(embedding, self.reference_embedding)

        is_match = similarity >= self.threshold

        return is_match, similarity

    def enroll_sample(self, audio: np.ndarray) -> bool:
        """
        Add an audio sample to the enrollment set.

        Args:
            audio: Audio sample of the user's voice

        Returns:
            True if enrollment was successful
        """
        if not self.enabled:
            return False

        embedding = self.extract_embedding(audio)
        if embedding is None:
            return False

        self.enrolled_embeddings.append(embedding)
        print(f"Enrolled sample {len(self.enrolled_embeddings)}/{self.min_enrollment_samples}")

        # Update reference embedding (average of all enrolled samples)
        self._update_reference_embedding()

        return True

    def _update_reference_embedding(self):
        """Update the reference embedding by averaging all enrolled samples."""
        if len(self.enrolled_embeddings) == 0:
            self.reference_embedding = None
        else:
            self.reference_embedding = np.mean(
                np.array(self.enrolled_embeddings), axis=0
            )

    def is_enrolled(self) -> bool:
        """Check if user has completed voice enrollment."""
        return len(self.enrolled_embeddings) >= self.min_enrollment_samples

    def save_profile(self):
        """Save the voice profile to disk."""
        if len(self.enrolled_embeddings) == 0:
            print("No enrollment data to save")
            return

        profile_data = {
            'embeddings': self.enrolled_embeddings,
            'reference': self.reference_embedding,
            'threshold': self.threshold,
        }

        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, 'wb') as f:
            pickle.dump(profile_data, f)

        print(f"Voice profile saved to {self.profile_path}")

    def load_profile(self) -> bool:
        """Load existing voice profile from disk."""
        source_path = self.profile_path
        if not source_path.exists() and self.legacy_profile_path.exists():
            source_path = self.legacy_profile_path

        if not source_path.exists():
            print("No existing voice profile found")
            return False

        try:
            with open(source_path, 'rb') as f:
                profile_data = pickle.load(f)

            self.enrolled_embeddings = profile_data['embeddings']
            self.reference_embedding = profile_data['reference']
            if source_path != self.profile_path:
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, self.profile_path)
                print(f"Migrated voice profile to {self.profile_path}")
            print(f"Voice profile loaded: {len(self.enrolled_embeddings)} samples")
            return True

        except Exception as e:
            print(f"Error loading voice profile: {e}")
            return False

    def clear_profile(self):
        """Clear the current voice profile."""
        self.enrolled_embeddings = []
        self.reference_embedding = None

        if self.profile_path.exists():
            self.profile_path.unlink()

        print("Voice profile cleared")

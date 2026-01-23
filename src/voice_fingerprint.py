"""
Voice fingerprinting module for Bloviate.
Uses speaker embeddings to verify that audio matches the enrolled user's voice.
"""

import numpy as np
import torch
import torchaudio

# Apply compatibility patch for newer torchaudio versions
import torchaudio_patch

from speechbrain.pretrained import EncoderClassifier
from typing import Optional, List
import os
import pickle
from pathlib import Path


class VoiceFingerprint:
    """
    Speaker verification using voice embeddings.
    Only accepts audio matching the enrolled voice profile.
    """

    def __init__(self, config: dict, model_dir: str = "models"):
        self.config = config
        self.enabled = config['voice_fingerprint']['enabled']
        self.threshold = config['voice_fingerprint']['threshold']
        self.sample_rate = config['audio']['sample_rate']
        self.model_name = config['voice_fingerprint']['embedding_model']
        self.min_enrollment_samples = config['voice_fingerprint']['min_enrollment_samples']

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.profile_path = self.model_dir / "voice_profile.pkl"

        # Load speaker embedding model
        print("Loading speaker embedding model...")
        try:
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

        with open(self.profile_path, 'wb') as f:
            pickle.dump(profile_data, f)

        print(f"Voice profile saved to {self.profile_path}")

    def load_profile(self) -> bool:
        """Load existing voice profile from disk."""
        if not self.profile_path.exists():
            print("No existing voice profile found")
            return False

        try:
            with open(self.profile_path, 'rb') as f:
                profile_data = pickle.load(f)

            self.enrolled_embeddings = profile_data['embeddings']
            self.reference_embedding = profile_data['reference']
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

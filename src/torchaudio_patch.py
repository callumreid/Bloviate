"""
Patch for torchaudio compatibility with speechbrain.
The list_audio_backends() function was removed in newer torchaudio versions.
"""

import torchaudio

# Monkey-patch the missing function
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        """Compatibility shim for newer torchaudio versions."""
        return ["soundfile"]  # Default backend in newer versions

    torchaudio.list_audio_backends = list_audio_backends

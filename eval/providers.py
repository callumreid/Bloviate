"""STT provider adapters for the eval harness.

Cloud providers read keys from the same env vars the app uses:
  DEEPGRAM_API_KEY, OPENROUTER_API_KEY
Nothing here logs a key. OpenRouter exposes many STT models behind one
OpenAI-compatible multipart endpoint, so any model id from
`/api/v1/models?output_modalities=transcription` can be passed straight through.
"""

import os
import json
import uuid
import urllib.parse
import urllib.request

# Terms worth biasing the interim pass toward (Deepgram keyterms). The final
# OpenRouter pass ignores prompts, so bias only reaches Deepgram.
KEYTERMS = [
    "bronson", "Coval", "Bloviate", "Callum", "Kappi", "Kobi", "Kobi Hudson",
    "Deepgram", "Pipecat", "Vapi", "LiveKit", "kubectl", "pytest", "ruff",
    "git checkout", "Slack", "standup", "Grok", "Dana",
]


def deepgram(path: str, model: str = "nova-3", keyterms: bool = False) -> str:
    params = [("model", model), ("smart_format", "true"), ("punctuate", "true")]
    if keyterms:
        params += [("keyterm", k) for k in KEYTERMS]
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, data=open(path, "rb").read(), method="POST",
        headers={"Authorization": "Token " + os.environ["DEEPGRAM_API_KEY"],
                 "Content-Type": "audio/wav"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d["results"]["channels"][0]["alternatives"][0]["transcript"]


def openrouter(path: str, model: str) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode()
    body += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/audio/transcriptions", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("text", "")


def mlx_local(path: str, repo: str = "mlx-community/whisper-large-v3-turbo") -> str:
    import mlx_whisper
    return mlx_whisper.transcribe(path, path_or_hf_repo=repo).get("text", "").strip()

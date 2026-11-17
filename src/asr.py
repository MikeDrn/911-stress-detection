"""Speech-to-text, fully offline, with faster-whisper (CTranslate2 Whisper).

Whisper turns the caller's audio into the transcript that the content model reads.
The default ``base.en`` model (~140 MB) runs on a laptop CPU in roughly 1-2 s per
15 s call (int8). In Docker the weights are downloaded once at *image build* time
("bake the model into the image", Course 6), so the container never needs the
internet at runtime.

Speech-to-text is *infrastructure*, versioned with the image (code release train);
the voice and content models are versioned in the MLflow registry (model train).
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

log = logging.getLogger("asr")


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str: ...


class WhisperTranscriber:
    def __init__(self, model: str = "base.en", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper/{model}"
        self.model = WhisperModel(
            model,
            device="cpu",
            compute_type=compute_type,
            download_root=os.environ.get("ASR_CACHE_DIR"),
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            raise ValueError("Whisper expects 16 kHz audio")
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class StaticTranscriber:
    """Test double / offline fallback: returns a fixed transcript."""

    def __init__(self, text: str = "", name: str = "static"):
        self.text, self.name = text, name

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self.text


def load_transcriber(spec: str | None) -> Transcriber | None:
    """``spec``: a Whisper model name (base.en, tiny.en, small.en ...) or 'none'."""
    if not spec or spec.lower() == "none":
        return None
    return WhisperTranscriber(spec)


def main() -> int:
    """`uv run python -m src.asr path.wav` - quick check that speech-to-text works."""
    import sys

    from src.preprocessing import load_audio

    model = os.environ.get("ASR_MODEL", "base.en")
    print(f"[asr] loading {model} (first run downloads the weights) ...")
    t = WhisperTranscriber(model)
    for path in sys.argv[1:]:
        print(f"[asr] {path}: {t.transcribe(load_audio(path))!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

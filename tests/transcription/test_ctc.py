import pytest

from grammar_scoring.transcription.ctc import (
    CTCConfig,
    normalize_ctc_text,
    transcript_record,
)


def test_normalize_ctc_text():
    assert normalize_ctc_text("  I LIKED TO  PLAYGROUND\tHE DON'T ") == (
        "i liked to playground he don't"
    )
    assert normalize_ctc_text("") == ""


def test_transcript_record_is_canonical_compatible():
    record = transcript_record("test", "audio_1.wav", "A LOT KIDS", 52.5)
    assert record == {
        "split": "test",
        "filename": "audio_1.wav",
        "text": "a lot kids",
        "language": "en",
        "duration_seconds": 52.5,
        "segments": [],
    }


def test_transcript_record_rejects_unknown_split():
    with pytest.raises(ValueError, match="Invalid split"):
        transcript_record("valid", "audio_1.wav", "x", 1.0)


def test_default_config():
    config = CTCConfig()
    assert config.model_name == "facebook/wav2vec2-large-960h-lv60-self"
    assert config.sampling_rate == 16_000
    assert config.stride_length_s < config.chunk_length_s / 2

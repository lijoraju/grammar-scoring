"""Create E020 CTC (wav2vec2, no language model) transcripts for train and test."""

from grammar_scoring.transcription.ctc import main

if __name__ == "__main__":
    main()

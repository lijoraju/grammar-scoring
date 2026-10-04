"""Generate parity-gated E006a and fixed E005/E006a test candidates offline."""

import json

from grammar_scoring.inference.acoustic_ensemble import run_acoustic_inference

if __name__ == "__main__":
    print(json.dumps(run_acoustic_inference(), indent=2))

"""E017: label-free explicit grammar signals from transcripts.

Two families of features are computed for every transcript, without labels:

* Grammatical error correction (GEC) edit rate: each sentence is corrected by
  a seq2seq GEC model (CoEdIT) and the word-level edit distance between the
  original and the correction is normalized by the number of words.
* Zero-shot LLM rubric judgement: an instruction-tuned LLM reads the
  competition rubric and the transcript; the probabilities of the answer
  tokens "1".."5" give an expected score and its uncertainty.

This module depends only on third-party packages so it can run as a Kaggle
script. Torch and Transformers are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

GEC_MODEL = "grammarly/coedit-large"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

RUBRIC = """Grammar score rubric for spoken English:
1: Struggles with sentence structure and syntax; limited control of simple \
grammatical structures and memorized patterns.
2: Limited understanding of sentence structure; consistently makes basic \
mistakes; may leave sentences incomplete.
3: Decent grasp of sentence structure but errors in grammatical structure, or \
vice versa.
4: Strong understanding of sentence structure and syntax; good control of \
grammar; occasional minor errors that do not cause misunderstanding.
5: High grammatical accuracy and adept control of complex grammar; seldom \
makes noticeable mistakes; handles complex structures well."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+|\d+")


def split_sentences(text: str, max_words: int = 40) -> list[str]:
    """Split ASR text into sentences, chunking unpunctuated run-ons.

    Args:
        text: Transcript text.
        max_words: Maximum words per chunk for long unpunctuated spans.

    Returns:
        Non-empty sentence strings.
    """
    sentences = []
    for sentence in _SENTENCE_END.split(text.strip()):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            chunk = " ".join(words[start : start + max_words])
            if chunk:
                sentences.append(chunk)
    return sentences


def words(text: str) -> list[str]:
    """Return lower-cased word tokens without punctuation."""
    return [token.lower() for token in _WORD.findall(text)]


def word_edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    """Compute the Levenshtein distance between two token sequences.

    Args:
        source: Original tokens.
        target: Corrected tokens.

    Returns:
        Minimum number of token insertions, deletions and substitutions.
    """
    previous = list(range(len(target) + 1))
    for i, token in enumerate(source, start=1):
        current = [i]
        for j, other in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token != other),
                )
            )
        previous = current
    return previous[-1]


def edit_features(originals: Sequence[str], corrections: Sequence[str]) -> dict:
    """Summarize GEC edits of one transcript's sentences.

    Args:
        originals: Original sentences.
        corrections: GEC outputs aligned with ``originals``.

    Returns:
        ``gec_edit_rate`` (edits per original word), ``gec_changed_fraction``
        (fraction of sentences changed) and ``gec_n_words``.
    """
    edits = n_words = changed = 0
    for original, corrected in zip(originals, corrections, strict=True):
        source, target = words(original), words(corrected)
        distance = word_edit_distance(source, target)
        edits += distance
        n_words += len(source)
        changed += distance > 0
    return {
        "gec_edit_rate": edits / max(n_words, 1),
        "gec_changed_fraction": changed / max(len(originals), 1),
        "gec_n_words": n_words,
    }


def expected_score(logits: np.ndarray) -> tuple[float, float]:
    """Convert logits over the answers "1".."5" into mean and std of the score.

    Args:
        logits: Array of five logits, for scores 1 to 5.

    Returns:
        ``(expected score, standard deviation)`` under the softmax distribution.
    """
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    scores = np.arange(1, 6)
    mean = float(probabilities @ scores)
    return mean, float(np.sqrt(probabilities @ (scores - mean) ** 2))


def gec_features(texts: Sequence[str], batch_size: int = 32) -> pd.DataFrame:
    """Run CoEdIT grammar correction and summarize edits per transcript."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        GEC_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    sentences = [split_sentences(text) for text in texts]
    flat = [s for group in sentences for s in group]
    corrected: list[str] = []
    for start in range(0, len(flat), batch_size):
        prompts = [
            f"Fix grammatical errors in this sentence: {s}"
            for s in flat[start : start + batch_size]
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to("cuda")
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=128, num_beams=1)
        corrected += tokenizer.batch_decode(output, skip_special_tokens=True)
    rows, offset = [], 0
    for group in sentences:
        rows.append(edit_features(group, corrected[offset : offset + len(group)]))
        offset += len(group)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def judge_features(texts: Sequence[str]) -> pd.DataFrame:
    """Score transcripts zero-shot with an instruction LLM and the rubric."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.eval()
    answer_ids = [
        tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)
    ]
    rows = []
    for text in texts:
        messages = [
            {
                "role": "system",
                "content": "You are an expert examiner of spoken English grammar.",
            },
            {
                "role": "user",
                "content": f"{RUBRIC}\n\nThe following is an automatic transcript "
                f"of a learner's spoken response.\n\nTranscript: {text}\n\n"
                "Rate the speaker's grammar from 1 to 5. Answer with one digit.",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**encoded).logits[0, -1, answer_ids].float().cpu().numpy()
        mean, std = expected_score(logits)
        rows.append({"judge_score": mean, "judge_std": std})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    """Compute E017 features for train and test transcripts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "test"):
        with (args.transcript_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "filename": [r["filename"] for r in records],
                    "text": [str(r["text"]).strip() for r in records],
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    gec = gec_features(frame["text"].tolist())
    print("GEC done", gec.describe().to_dict(), flush=True)
    judge = judge_features(frame["text"].tolist())
    print("judge done", judge.describe().to_dict(), flush=True)
    features = pd.concat([frame[["split", "filename"]], gec, judge], axis=1)
    features.to_csv(args.output_dir / "grammar_signals.csv", index=False)
    (args.output_dir / "grammar_signals.meta.json").write_text(
        json.dumps({"gec_model": GEC_MODEL, "judge_model": JUDGE_MODEL})
    )


if __name__ == "__main__":
    main()

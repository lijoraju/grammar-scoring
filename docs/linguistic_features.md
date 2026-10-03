# E004 linguistic feature extraction

This stage extracts 93 numeric features from canonical raw Whisper transcripts.
It does not access labels, folds, embeddings, or predictions. No real extraction
or modeling is part of this implementation. Schema version: `1`.

## Installation and first extraction

spaCy is an optional `linguistic` experiment dependency, also included in the dev
group because offline tests construct spaCy Docs. Core installations do not need
spaCy. The small English model is a separate asset and is never committed.

```sh
uv sync --extra linguistic
uv run --extra linguistic python -m spacy download en_core_web_sm
uv run --extra linguistic python scripts/generate_linguistic_features.py --split all --transcript-dir artifacts/transcripts --output-dir artifacts/features --batch-size 32
```

Tests never download or load that model. Installation chooses the compatible small
model; extraction metadata records its actual version. Only `en_core_web_sm` is
loaded, with parser/tagger components enabled. `--overwrite` is required if any
requested CSV or the shared metadata exists. When adding a split to an existing
output directory, use `--overwrite`; configurations must match any retained split.

## Definitions

- Words: spaCy tokens that are neither space nor punctuation. Numbers and other
  non-punctuation tokens count. Lexical types use casefolded token text, without
  lemmatization. Raw text passed into spaCy is unchanged.
- TTR: unique words / words. Root TTR: unique words / sqrt(words).
- Sentence statistics include every spaCy sentence, including punctuation-only
  sentences. Standard deviation is the population statistic. Mean word length
  counts token characters. Character count includes all raw whitespace and
  punctuation. Punctuation count counts spaCy punctuation tokens; question and
  exclamation counts count literal characters in raw text.
- Duration is canonical transcript duration. All zero-denominator ratios are 0;
  empty documents produce 0 statistics, with duration retained when nonzero.
- Finite verbs: VERB or AUX with `VerbForm=Fin`. Past/present verbs: VERB or AUX
  with `Tense=Past` / `Tense=Pres`, independently of finiteness. Modal auxiliaries:
  POS AUX and fine tag MD. Missing morphology contributes no matches.
- Dependency distance: absolute spaCy token-index difference for non-root arcs
  whose dependent and head are linguistic words. Punctuation positions remain
  part of index distance. Depth: number of ancestors of each linguistic token,
  root depth 0; intermediate ancestors are not filtered. Depth statistics aggregate
  tokens across the document, not sentence averages. Missing POS/dependency
  annotations on nonempty documents cause an error rather than guessed features.
- Nominal subjects: nsubj. Passive subjects: nsubjpass/csubjpass. Objects:
  dobj/obj/iobj/pobj/dative. Clausal subjects: csubj/csubjpass. Clausal complements:
  ccomp. Open complements: xcomp. Adverbial clauses: advcl. Relative clauses:
  relcl. Coordination: conj. Indicators count matching linguistic tokens and divide
  by sentence count for per-sentence variants. A csubjpass contributes to both
  passive-subject and clausal-subject indicators; each resulting feature has one
  family membership.
- Sentence subject presence: any nsubj/nsubjpass/csubj/csubjpass token. Finite verb
  presence follows the morphology rule above. Completeness is the fraction of
  sentences with both; it is a proxy, not a grammar correctness judgment.
- Very short sentences have <= 3 linguistic words, including zero-word sentences.
- Repetition uses casefolded linguistic token text with punctuation removed.
  Adjacent repeats count matching neighboring pairs; rate divides by max(words-1,
  0). N-grams use overlapping document-wide windows, including across sentence
  boundaries. Repeated bigram/trigram counts sum frequency minus one per distinct
  n-gram; rates divide by all eligible windows. Sentence repeats compare complete
  linguistic token tuples and count excess occurrences; rate divides by sentences.
- Fillers: uh, um, hmm, er, ah, you know, i mean. Count left-to-right nonoverlapping
  token-sequence matches within each sentence, ignoring punctuation/case. Each
  expression counts once; multiword matches consume their words. Matches do not
  cross sentence boundaries. `like` has no filler status. Per-100-word rates use
  all linguistic words, including filler words.

## APIs and artifacts

`features.linguistic` exposes `FEATURE_FAMILIES`, `FEATURE_COLUMNS`, four family
extractors, `extract_document(split, filename, text, duration_seconds, doc)`, and
`extract_frame(transcripts, nlp, batch_size=32)`. The injected pipeline is reused
via `nlp.pipe`; output preserves canonical JSONL order. Family and column ordering
are explicit. Identities are not feature members.

`features.linguistic_artifacts` exposes `validate_features`, `save_features`, and
`load_features`. Validation requires the exact ordered schema, finite numeric
features, valid unique identities, expected split, and exact canonical transcript
identity ordering. CSV headers are checked before pandas can rename duplicates.

The thin script delegates to `features.linguistic_cli.generate_main`. It loads the
model once, processes splits separately, and writes `linguistic_train.csv`,
`linguistic_test.csv`, and `linguistic_features.meta.json`. CSVs contain only split,
filename, and features. Metadata includes schema/model/Python versions, registry,
ordered columns, threshold, filler lexicon, batch size, sample counts, and SHA-256
source/output hashes. Extraction is deterministic with no sampling or random seed.

## Exact ordered feature registry

### surface (19)

```text
word_count
unique_word_count
type_token_ratio
root_type_token_ratio
sentence_count
mean_sentence_length
median_sentence_length
std_sentence_length
min_sentence_length
max_sentence_length
character_count
mean_word_length
punctuation_count
punctuation_per_100_words
question_mark_count
exclamation_mark_count
duration_seconds
words_per_second
words_per_minute
```

### pos (34)

```text
pos_noun_count
pos_noun_per_100_words
pos_propn_count
pos_propn_per_100_words
pos_verb_count
pos_verb_per_100_words
pos_aux_count
pos_aux_per_100_words
pos_adj_count
pos_adj_per_100_words
pos_adv_count
pos_adv_per_100_words
pos_pron_count
pos_pron_per_100_words
pos_det_count
pos_det_per_100_words
pos_adp_count
pos_adp_per_100_words
pos_cconj_count
pos_cconj_per_100_words
pos_sconj_count
pos_sconj_per_100_words
pos_part_count
pos_part_per_100_words
pos_num_count
pos_num_per_100_words
pos_intj_count
pos_intj_per_100_words
finite_verb_count
finite_verbs_per_100_words
past_verb_count
present_verb_count
modal_auxiliary_count
modal_auxiliaries_per_100_words
```

### syntax (28)

```text
mean_dependency_distance
max_dependency_distance
mean_dependency_tree_depth
max_dependency_tree_depth
nominal_subject_count
nominal_subject_per_sentence
passive_subject_count
passive_subject_per_sentence
object_count
object_per_sentence
clausal_subject_count
clausal_subject_per_sentence
clausal_complement_count
clausal_complement_per_sentence
open_clausal_complement_count
open_clausal_complement_per_sentence
adverbial_clause_count
adverbial_clause_per_sentence
relative_clause_count
relative_clause_per_sentence
coordination_count
coordination_per_sentence
sentences_with_subject_count
sentences_with_finite_verb_count
sentences_with_subject_and_finite_verb_count
sentences_with_subject_rate
sentences_with_finite_verb_rate
sentence_completeness_rate
```

### spoken (12)

```text
very_short_sentence_count
very_short_sentence_rate
adjacent_repeated_word_count
adjacent_repeated_word_rate
repeated_bigram_count
repeated_bigram_rate
repeated_trigram_count
repeated_trigram_rate
repeated_sentence_count
repeated_sentence_rate
filler_count
fillers_per_100_words
```

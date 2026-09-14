# TFP-FND: Time-Frequency-Propagation Framework for Adversarially Robust Fake News Detection

Reference implementation of the architecture and experimental protocol described in
*"TFP-FND: A Time-Frequency-Propagation Framework for Adversarially Robust Fake News
Detection"*, built to reproduce Sections 3-5 of the paper (methodology, experimental
setup, results/RQ1-RQ6).

> **Honesty note.** This repository faithfully implements the described architecture,
> attack generators, preprocessing pipeline, and evaluation protocol. It does **not**
> ship pretrained checkpoints or guarantee it will reproduce the exact numbers already
> printed in the paper's tables — those depend on your data, hardware, seeds, and any
> hyperparameter tuning you do. Treat the numbers in the paper as targets to work
> towards, not as something this code will output automatically on first run.

## 1. Repository layout

```
tfp-fnd/
├── configs/                  # one YAML per dataset, mirrors Table "Selected training hyperparameters"
│   ├── pheme.yaml
│   ├── twitter15.yaml
│   └── twitter16.yaml
├── data/
│   ├── raw/                  # <- put the official PHEME / Twitter15 / Twitter16 releases here (see §3)
│   │   ├── pheme/
│   │   ├── twitter15/
│   │   └── twitter16/
│   └── processed/            # written by the preprocessing scripts
├── src/
│   ├── preprocessing/        # §3.5 dataset construction (PHEME audit table, Twitter15/16 mapping)
│   ├── data/                 # PyTorch Dataset / collate for the 4-condition (clean, A_c, A_p, A_cp) batches
│   ├── models/               # content / temporal / propagation views + adaptive fusion + baselines
│   ├── attacks/               # content attack (LLM paraphrase + NLI filter) and propagation attack generator
│   ├── training/              # losses (L_cls, L_robust, L_view), train loop, evaluation + RQ scripts
│   └── utils/                 # seeding, metrics (Macro-F1, MCC, degradation, retention ratio)
├── scripts/                   # shell entry points for each experiment in the paper
└── results/                   # CSV/JSON outputs land here
```

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI key if you want to regenerate content attacks (`A_c`, `A_cp`) rather than
reuse cached ones:

```bash
export OPENAI_API_KEY=sk-...
```

## 3. Persisting work across sessions (Colab + Google Drive)

If you're running this in Google Colab: `/content` is wiped on every disconnect, so
anything expensive to regenerate — preprocessed data, generated attacks, checkpoints,
results — should live on Drive, not in `/content` directly. Keep the *raw* dataset
archives (many small files once extracted) on local disk each session for speed, but
symlink the *outputs* (few, larger files) to Drive so they persist automatically as
scripts write them, with no manual zip/copy step at the end:

```bash
from google.colab import drive
drive.mount('/content/drive')

WORKSPACE = "/content/drive/MyDrive/tfp-fnd-workspace"
!mkdir -p {WORKSPACE}/raw_archives {WORKSPACE}/data_processed {WORKSPACE}/checkpoints {WORKSPACE}/results

%cd /content/tfp-fnd
!rm -rf data/processed checkpoints results
!ln -s {WORKSPACE}/data_processed data/processed
!ln -s {WORKSPACE}/checkpoints checkpoints
!ln -s {WORKSPACE}/results results
```

Upload `PHEME_veracity_tar.bz2` / `rumdetect2017_2_.zip` into
`{WORKSPACE}/raw_archives/` once (via the Colab file browser or `drive.mount` + a normal
`cp`) and extract them into local `/content` scratch space every session — extracting
straight onto a Drive-mounted path is slow (PHEME alone is 100k+ small JSON files over
FUSE); the processed JSONL/checkpoint files preprocessing produces are far fewer and
larger, so those are fine to write directly to Drive. On a fresh session, re-run step 3
below to pull the raw archives back from Drive and re-extract locally, then skip straight
to step 6 onward if `data/processed/<dataset>/instances.jsonl` already exists on Drive.

## 4. Add your data

Drop the official releases in `data/raw/<dataset>/` exactly as distributed upstream:

- **PHEME** (v1.0, DOI 10.15127/1.293723): unzip so that
  `data/raw/pheme/<event>/{rumours,non-rumours}/<thread_id>/{source-tweet,reactions,structure.json}`
  exists for each of the 9 events.
- **Twitter15 / Twitter16** (Ma et al. 2016 release): unzip so that
  `data/raw/twitter15/{source_tweets.txt,label.txt,tree/}` and the Twitter16 equivalent exist.

Nothing under `data/raw/` is committed — see `.gitignore`.

## 5. Preprocess

```bash
# PHEME: applies the audit pipeline in Table "PHEME preprocessing audit"
python -m src.preprocessing.pheme_preprocess \
    --raw_dir data/raw/pheme --out_dir data/processed/pheme

# Twitter15 / Twitter16: conservative true/false-only mapping
python -m src.preprocessing.twitter_preprocess \
    --raw_dir data/raw/twitter15 --dataset_name twitter15 --out_dir data/processed/twitter15
python -m src.preprocessing.twitter_preprocess \
    --raw_dir data/raw/twitter16 --dataset_name twitter16 --out_dir data/processed/twitter16

# Event-disjoint PHEME splits (Table "Event-disjoint splits for PHEME") +
# stratified random splits for Twitter15/16
python -m src.preprocessing.splits --dataset pheme --processed_dir data/processed/pheme
python -m src.preprocessing.splits --dataset twitter15 --processed_dir data/processed/twitter15
python -m src.preprocessing.splits --dataset twitter16 --processed_dir data/processed/twitter16
```

Each preprocessing run writes a `preprocessing_audit.json` next to its output so the
counts in your run can be diffed against the paper's Table 4 (PHEME) — they should match
if you're using the same v1.0 PHEME release; if they don't, the audit file tells you
exactly which filter stage diverged.

## 6. Generate attacks

```bash
# Propagation attacks (no LLM calls, deterministic given --seed)
python -m src.attacks.propagation_attack \
    --processed_dir data/processed/pheme --split_dir data/processed/pheme/splits \
    --kappa 0.10 0.20 0.30 0.50 --seed 42

# Content attacks (calls OpenAI; cache results so you only pay for generation once)
python -m src.attacks.content_attack \
    --processed_dir data/processed/pheme --model gpt-4o-2024-08-06 \
    --temperature 0.7 --top_p 0.9 --max_tokens 200 --num_candidates 5 \
    --nli_model microsoft/deberta-large-mnli --entail_thresh 0.85 --contra_thresh 0.15
```

**Model availability.** The paper's original design specifies `gpt-4-0613`. OpenAI
periodically retires old dated model snapshots, and as of this codebase's last update
that one is no longer available -- `gpt-4o-2024-08-06` (a dated, pinned GPT-4o snapshot,
not a moving `-latest` alias) is used as the default substitute. If you hit a `Model
'...' is not available` error, run this to see what your account currently has access
to, then pass `--model <name>` explicitly:
```python
from openai import OpenAI
print(sorted(m.id for m in OpenAI().models.list().data if "gpt" in m.id.lower()))
```
Whatever model you actually use should be stated explicitly in the paper's methodology
section, since it materially affects the generated attack samples -- don't silently
treat this default as equivalent to the originally-specified `gpt-4-0613`.

Both scripts respect the paper's leakage-prevention rule: attacks are generated **after**
split assignment, and every perturbed sample inherits its source item's partition.

## 7. Train

```bash
python -m src.training.train --config configs/pheme.yaml --split 1 --model tfp_fnd --seed 0
```

`--model` accepts `tfp_fnd`, `tfp_fnd_wo_robust`, `bert_base`, `bert_dft`, `bert_temporal`,
`bert_bigcn`, `bert_hgsl`, or an ablation flag combination (see `configs/pheme.yaml` for the
full list and `src/models/baselines.py` for what each name builds). Bi-GCN, HG-SL, and LeHT
are provided as best-effort reimplementations from their papers' architectural
descriptions (§4.6 of the paper / `src/models/baselines.py` docstrings) — verify against
the authors' own released code if you need a citable exact match.

## 8. Reproduce the paper's RQs

```bash
scripts/run_main_experiment.sh      # Table "main_results" (RQ1) — all baselines x 5 PHEME splits
scripts/run_ablation.sh             # Table "ablation_arch" + "robustness_ablation" (RQ2, RQ6)
scripts/run_attack_family_sweep.sh  # Table "attack_families" (RQ3)
scripts/run_early_detection.sh      # Table "early_deadline" (RQ5), horizons 5m/10m/20m/30m/45m/60m/24h
scripts/run_attack_sensitivity.sh   # Table "attack_sensitivity" — kappa/rho3/w/gamma1 sweeps
scripts/run_cross_dataset.sh        # Table "twitter_results"
```

Each script writes a CSV to `results/` with per-seed, per-split rows plus a
`aggregate.py`-generated mean±std summary matching the tables' format, and (for the main
experiment) a paired significance test between TFP-FND and the strongest baseline per
split, addressing the "Statistical Evaluation Protocol" section of the paper.

## 9. Known gaps / TODO

- LeHT is approximated (the original paper's exact hyperparameters for its LLM-content
  enrichment step aren't fully specified in its published version); check
  `src/models/baselines.py::LeHTApprox` before citing head-to-head numbers against it.
- The content-attack generator caches by `(source_id, model, temperature)` so repeated
  runs don't re-spend API budget; delete `data/processed/<dataset>/attack_cache/` to force
  regeneration.
- No repository code decides what counts as a "final" number for the paper — you still
  need to run the seeds/splits, aggregate, and sanity-check before reporting.

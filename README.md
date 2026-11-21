# TFP-FND: Time-Frequency-Propagation Framework for Adversarially Robust Fake News Detection

Reference implementation of **TFP-FND**, a time-, frequency-, and propagation-aware framework for adversarially robust fake-news detection.

This repository contains the implementation, experiment scripts, and a trained checkpoint for the PHEME configuration (`split1_seed0`). The supplied checkpoint is intended to make evaluation/reproduction easier without requiring the user to train the model from scratch.

> **Dataset note:** PHEME, Twitter15, and Twitter16 are third-party datasets. They are **not redistributed in this repository**. Please obtain them from their original sources and cite the original dataset papers.

## Repository structure

```text
tfp-fnd/
├── configs/                  # Dataset/model configuration files
├── data/
│   ├── raw/                 # Put the original datasets here (not included)
│   └── processed/           # Generated preprocessing outputs (not included)
├── checkpoints/
│   └── pheme/
│       └── tfp_fnd/
│           └── split1_seed0/
│               └── best.pt  # Supplied trained checkpoint
├── src/
│   ├── preprocessing/       # Dataset preprocessing and split generation
│   ├── data/                # Dataset, collate and historical context utilities
│   ├── models/              # TFP-FND and baseline models
│   ├── attacks/             # Content and propagation attacks
│   ├── training/            # Training and evaluation
│   └── utils/               # Seeds and evaluation metrics
├── scripts/                 # Experiment entry points
├── results/                 # Experiment outputs
├── requirements.txt
└── README.md
```

## Dataset availability

The original datasets are available from their respective sources:

- **PHEME – Rumour Detection and Veracity Classification:**  
  https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078

- **Twitter15 and Twitter16:**  
  https://github.com/gszswork/Twitter15_16_dataset

The repository does not claim ownership of these datasets. Users should consult the original sources for licensing, attribution, and data-use requirements and cite the original dataset publications.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

Place the original datasets in the following locations:

```text
data/raw/pheme/
data/raw/twitter15/
data/raw/twitter16/
```

Then run preprocessing:

```bash
python -m src.preprocessing.pheme_preprocess \
    --raw_dir data/raw/pheme \
    --out_dir data/processed/pheme

python -m src.preprocessing.twitter_preprocess \
    --raw_dir data/raw/twitter15 \
    --dataset_name twitter15 \
    --out_dir data/processed/twitter15

python -m src.preprocessing.twitter_preprocess \
    --raw_dir data/raw/twitter16 \
    --dataset_name twitter16 \
    --out_dir data/processed/twitter16
```

Generate the splits:

```bash
python -m src.preprocessing.splits \
    --dataset pheme \
    --processed_dir data/processed/pheme

python -m src.preprocessing.splits \
    --dataset twitter15 \
    --processed_dir data/processed/twitter15

python -m src.preprocessing.splits \
    --dataset twitter16 \
    --processed_dir data/processed/twitter16
```

## Checkpoint

A trained TFP-FND checkpoint is included:

```text
checkpoints/pheme/tfp_fnd/split1_seed0/best.pt
```

It corresponds to:

```text
Dataset : PHEME
Model   : TFP-FND
Split   : 1
Seed    : 0
File    : best.pt
```

### Using the checkpoint

The checkpoint can be used to reproduce the evaluation of the supplied trained model once the corresponding PHEME data have been downloaded and preprocessed.

The training code saves checkpoints using the same convention:

```text
checkpoints/<dataset>/<model>/split<split>_seed<seed>/best.pt
```

To train the same configuration from scratch:

```bash
python -m src.training.train \
    --config configs/pheme.yaml \
    --split 1 \
    --model tfp_fnd \
    --seed 0
```

## GitHub and large checkpoint files

The supplied `best.pt` file is approximately 430 MB. GitHub's normal Git file limit is 100 MB, so **do not upload this file with ordinary `git add`**.

For a GitHub repository, use Git LFS:

```bash
git lfs install
git lfs track "checkpoints/**/*.pt"
git add .gitattributes
git add checkpoints/pheme/tfp_fnd/split1_seed0/best.pt
git commit -m "Add PHEME TFP-FND checkpoint"
git push
```

Alternatively, publish the checkpoint as a GitHub Release asset or on a research data repository and put the download link in this README. Git LFS or a release is preferable to committing a 430 MB binary directly to Git.

## Generate adversarial attacks

Propagation attacks:

```bash
python -m src.attacks.propagation_attack \
    --processed_dir data/processed/pheme \
    --split_dir data/processed/pheme/splits \
    --kappa 0.10 0.20 0.30 0.50 \
    --seed 42
```

Content attacks:

```bash
python -m src.attacks.content_attack \
    --processed_dir data/processed/pheme \
    --model gpt-4-0613 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 200 \
    --num_candidates 5 \
    --nli_model microsoft/deberta-large-mnli \
    --entail_thresh 0.85 \
    --contra_thresh 0.15
```

If content attacks are regenerated, an OpenAI API key is required:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

Do not commit API keys to GitHub.

## Main experiment

Train TFP-FND:

```bash
python -m src.training.train \
    --config configs/pheme.yaml \
    --split 1 \
    --model tfp_fnd \
    --seed 0
```

The training procedure writes the best checkpoint to:

```text
checkpoints/pheme/tfp_fnd/split1_seed0/best.pt
```

## Reproducing the experimental protocol

The repository provides scripts corresponding to the main experimental questions:

```bash
scripts/run_main_experiment.sh
scripts/run_ablation.sh
scripts/run_attack_family_sweep.sh
scripts/run_early_detection.sh
scripts/run_attack_sensitivity.sh
scripts/run_cross_dataset.sh
```

Results are written to `results/`.

For exact reproduction of reported numbers, use the same dataset versions, preprocessing decisions, model configuration, random seeds, attack-generation settings, and software dependencies reported in the accompanying paper.

## Reproducibility checklist

Before reporting results:

- [ ] Download the original PHEME/Twitter15/Twitter16 data.
- [ ] Verify the dataset version and provenance.
- [ ] Run the preprocessing scripts.
- [ ] Generate the required split files.
- [ ] Use the supplied `best.pt` checkpoint for the provided PHEME split, or retrain from scratch.
- [ ] Keep the random seed fixed when comparing runs.
- [ ] Record the software/dependency versions.
- [ ] For content attacks, record the exact model and generation parameters.
- [ ] Aggregate results over the required splits/seeds before comparing with the paper.

## Citation

If you use the TFP-FND implementation, please cite the associated paper.

When using PHEME, Twitter15, or Twitter16, also cite the **original dataset publications** and follow their respective data-use requirements.

## License and third-party data

The source code in this repository should be considered separately from the third-party datasets. The PHEME, Twitter15, and Twitter16 data remain subject to their original licenses, terms, and attribution requirements.

The repository does not grant additional rights to third-party dataset content.

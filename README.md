# RefactorAssist Replication Package

This repository is the replication package for RefactorAssist.

## Package Scope

The active package contains three components:

1. Pipeline implementation:
   - [feedback_loop](feedback_loop)
2. Dataset and repository metadata used by the pipeline:
   - [data/dataset.csv](data/dataset.csv)
   - [data/repositories.csv](data/repositories.csv)
   - [data/repository_stats.csv](data/repository_stats.csv)
3. Final results organized by research question:
   - [results](results)

## Repository Layout

- [feedback_loop](feedback_loop): RefactorAssist runtime and repair loop.
- [data](data): CSV inputs and repository metadata.
- [results/rq1](results/rq1): RQ1 pass-rate artifacts.
- [results/rq2](results/rq2): RQ2 failure-category artifact.
- [results/rq3](results/rq3): RQ3 feedback-loop artifacts.
- [results/rq4](results/rq4): RQ4 ablation artifacts.

## Running the Pipeline

Entry point:

- [feedback_loop/main.py](feedback_loop/main.py)

The pipeline supports two initial-refactor modes:

- generate: run the initial StarCoder refactor in the loop.
- precomputed: start from pre-generated iter1 files.

Example:

```bash
python feedback_loop/main.py \
  --cuda-device cuda:0 \
  --input-csv data/dataset.csv \
  --output-file results_new.csv \
  --initial-refactor-mode generate
```

## Input CSV Expectations

The pipeline expects rows with:

- repository identifier: repo or repository
- method path: method
- test path: test_method
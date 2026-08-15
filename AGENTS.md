# Instructions for experiment agents

Read [`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md) before changing or running the experiment pipeline.

## Non-negotiable rules

- New experiments use `scripts/*.py` and Hydra configs. The notebooks are historical records, not the active runner.
- Keep the shared protocol fixed unless the task explicitly changes it: `Qwen/Qwen2.5-0.5B`, seed 42, 200k selection pool, 90k selected rows, one target-training epoch, assistant-only loss, packing, and the common 4,500-row holdout.
- Never compare runs without checking dataset revision/fingerprint, common-eval fingerprint, selection SHA, training protocol SHA, and benchmark dataset revision.
- Never reuse an output directory for a different protocol. Use a new directory instead of deleting or bypassing lineage checks.
- Preserve the import order in GPU entrypoints: Unsloth must be imported before Transformers or TRL. Keep the associated `# noqa: I001` where needed.
- Preserve existing user data and Drive artifacts. Do not remove checkpoints, adapters, score caches, W&B runs, or unrelated worktree changes unless explicitly asked.
- Selection methods belong in `scripts/select_data.py`; `scripts/train.py` must remain method-agnostic and consume only a versioned selection artifact.
- Treat common-eval loss, external benchmark accuracy, document count, token count, optimizer steps, FLOPs, and runtime as different measurements. Fixed 90k-row runs are not compute-controlled when selected lengths differ.
- Run targeted Ruff, `pytest`, byte compilation, Hydra config composition, and a GPU smoke run before recommending a full multi-hour selection or training run.

## Current implementation status

- Implemented in the runner: random, IFD, entropy, and RHO-Loss.
- Historical notebook only: quality classifier. It must be ported into the runner before it is used in the current comparison.
- External benchmarks: public ruMMLU and public labeled MERA Core.
- Statistical comparison: paired bootstrap confidence interval and exact McNemar test via `scripts/compare_benchmarks.py`.

The runbook contains canonical Colab commands, artifact locations, resume behavior, current results, and the next experimental work.

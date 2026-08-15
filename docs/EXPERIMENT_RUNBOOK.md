# Experiment Agent Runbook

This document is the operational source of truth for running and extending the reproducible SFT data-selection experiments. It is written for coding agents continuing work in this repository and for agents guiding a Colab operator.

## 1. Objective and current scope

The experiment compares data-selection strategies under one target SFT protocol:

| Component | Fixed value |
|---|---|
| Training dataset | `d0rj/ru-instruct`, `train` split |
| Target model | `Qwen/Qwen2.5-0.5B` |
| Quantization | Unsloth 4-bit loading |
| Shuffle seed | 42 |
| Common holdout | first 4,500 shuffled rows |
| Candidate pool | next 200,000 shuffled rows |
| Selection budget | 90,000 rows |
| Target training | one epoch, QLoRA, packed, assistant-only loss |
| Effective batch | 4 per device x 2 accumulation = 8 |
| Learning rate | `2e-4`, cosine schedule, 3% warmup |
| Checkpoints | every 200 optimizer steps, keep at most 13 |
| W&B | project `diploma-sft`, group `diploma-rerun-05` |

Implemented selection methods:

- `random`: deterministic random 90k from the common 200k pool.
- `ifd`: instruction-following difficulty based on conditional/unconditional perplexity.
- `entropy`: top mean categorical entropy over supervised assistant tokens.
- `rho`: top reducible loss, `L_base - L_IL`, using a separately trained IL proxy.

The quality classifier exists only in `notebooks/selection_quality_classifier.ipynb`. Do not produce a new classifier result from that notebook and present it as part of the current runner protocol. Port it first.

## 2. Repository map

| Path | Responsibility |
|---|---|
| `configs/config.yaml` | Shared Hydra defaults, dataset, model, W&B, output interpolation |
| `configs/selection/*.yaml` | Method-specific selection parameters |
| `configs/train/qwen05b_qlora.yaml` | Fixed target-training protocol |
| `configs/benchmark/*.yaml` | ruMMLU and MERA Core protocols |
| `scripts/select_data.py` | Build a versioned selection artifact; this is where method choice happens |
| `scripts/train.py` | Train from selection indices; intentionally method-agnostic |
| `scripts/evaluate.py` | Common holdout plus deterministic selected-train audit |
| `scripts/benchmark.py` | Resumable public ruMMLU or MERA Core evaluation |
| `scripts/compare_benchmarks.py` | Paired bootstrap CI, exact McNemar, disagreement export |
| `diploma_sft/artifacts.py` | Artifact contracts, checksums, cache and lineage guards |
| `diploma_sft/data.py` | Dataset reconstruction, fixed layout, rendering, fingerprints |
| `diploma_sft/*_selection.py` | Pure method scoring/selection logic |
| `diploma_sft/evaluation.py` | Assistant-only loss and perplexity |
| `diploma_sft/rummlu.py` | ruMMLU prompt construction and aggregation |
| `diploma_sft/mera_core.py` | MERA task adapters and aggregation |
| `diploma_sft/paired_analysis.py` | Statistical paired comparison primitives |
| `tests/` | CPU-safe tests for contracts and pure logic |
| `notebooks/` | Historical experiments; not the active runner |

## 3. Data layout and leakage boundary

All methods start from the same immutable dataset revision and the same shuffle:

```text
ds_full.shuffle(seed=42)
├── [0:4,500)          common evaluation holdout
├── [4,500:204,500)    shared 200k selection/candidate pool
└── [204,500:234,500)  RHO-only D_ho, 30k rows
```

RHO splits its separate 30k `D_ho` deterministically into 95% IL train and 5% IL eval. It still ranks all rows in the same 200k candidate pool used by the other methods. This is intentionally different from the historical RHO notebook, which consumed part of its 200k pool for `D_ho` and therefore had a different candidate set.

Selected indices are always relative to the shared 200k pool. `scripts/train.py` reconstructs the original dataset at the resolved Hugging Face revision, repeats the shuffle, rebuilds the pool, and selects exactly those indices.

## 4. Colab setup

Choose a CUDA runtime with bf16 support before running the setup. The scripts stop instead of silently falling back to fp16.

```python
from google.colab import drive
drive.mount("/content/drive")
```

For an existing checkout:

```bash
%cd /content/diploma-public
!git switch codex/experiment-runner
!git pull --ff-only origin codex/experiment-runner
!pip install unsloth
!pip install -e ".[experiments]"
!wandb login
```

For a missing checkout:

```bash
%cd /content
!git clone -b codex/experiment-runner https://github.com/CHB-0r1s/diploma-public.git
%cd /content/diploma-public
!pip install unsloth
!pip install -e ".[experiments]"
!wandb login
```

Sanity checks:

```bash
!pwd
!git rev-parse --short HEAD
!python scripts/select_data.py --help
```

Expected working directory: `/content/diploma-public`. Do not run `pip install -e .` from `/content`.

## 5. Canonical output layout

Persistent artifacts belong on Google Drive:

```text
/content/drive/MyDrive/diploma/
├── selections/
│   ├── random_90000/
│   ├── ifd_qwen05b_90000/
│   ├── entropy_qwen05b_90000/
│   └── rho_qwen05b_90000/
├── outputs/
│   ├── baseline_random_qwen05b_90k/
│   ├── ifd_qwen05b_90k/
│   ├── entropy_qwen05b_90k/
│   └── rho_qwen05b_90k/
└── comparisons/
```

Do not reuse an existing directory name for changed model, seed, selection parameters, code protocol, or benchmark scope. Pick a new name. The runner rejects many incompatible resumes, but directory naming is still part of experimental hygiene.

## 6. Pipeline overview

Every method follows the same stage order:

```text
select -> train -> common eval + train audit -> ruMMLU -> MERA Core -> paired analysis
```

The method is chosen only during `select`. Training and evaluation receive the resulting `selection_manifest.json`.

### 6.1 Selection commands

Random:

```bash
!python scripts/select_data.py \
  selection=random \
  selection_output_dir=/content/drive/MyDrive/diploma/selections/random_90000
```

IFD:

```bash
!python scripts/select_data.py \
  selection=ifd \
  selection_output_dir=/content/drive/MyDrive/diploma/selections/ifd_qwen05b_90000
```

Entropy:

```bash
!python scripts/select_data.py \
  selection=entropy \
  selection_output_dir=/content/drive/MyDrive/diploma/selections/entropy_qwen05b_90000
```

RHO-Loss:

```bash
!python scripts/select_data.py \
  selection=rho \
  selection_output_dir=/content/drive/MyDrive/diploma/selections/rho_qwen05b_90000
```

Expected terminal output ends with the path to `selection_manifest.json`. Before training, inspect at least:

```bash
!python -m json.tool /content/drive/MyDrive/diploma/selections/rho_qwen05b_90000/selection_manifest.json | head -100
```

Method-specific resumable files:

| Method | Files |
|---|---|
| Random | `selected_indices.npy`, `selection_manifest.json` |
| IFD | `scores_cond.npy`, `scores_uncond.npy`, `scores_ifd.npy` |
| Entropy | `scores_entropy.npy` |
| RHO | `il_training/checkpoint-*`, `il_adapter/`, `scores_base.npy`, `scores_il.npy`, `scores_rho.npy` |

Repeating the exact same selection command continues compatible caches. Do not alter code, model, batch size, dataset revision, pool, or method parameters while reusing the directory. Cache compatibility is bound by `scoring_cache_manifest.json`.

### 6.2 Target training

Substitute `METHOD`, selection directory, and experiment name consistently:

```bash
!python scripts/train.py \
  experiment_name=rho_qwen05b_90k \
  selection_artifact=/content/drive/MyDrive/diploma/selections/rho_qwen05b_90000/selection_manifest.json \
  output_dir=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k
```

Training writes:

```text
outputs/<experiment>/
├── checkpoint-200/
├── checkpoint-400/
├── ...                         at most 13 latest checkpoints
├── adapter/                    final adapter used for evaluation
│   └── training_run_manifest.json
├── config.resolved.json
├── environment.json
├── dataset_metadata.json
├── selection_manifest.json
├── selected_indices.npy
├── training_run_manifest.json
└── train_result.json
```

`resume_from_checkpoint=auto` is the default. Repeating the identical command resumes the latest `checkpoint-*`. A checkpoint contains adapter weights plus optimizer, scheduler, RNG, and Trainer state. `adapter/` contains the final adapter after the full epoch and is the canonical evaluation input.

The runner does not select a best target checkpoint. It trains for the fixed one-epoch budget and saves the final adapter. Intermediate checkpoint evaluation is a separate experiment and must be named and reported as such.

### 6.3 Common evaluation and train audit

```bash
!python scripts/evaluate.py \
  experiment_name=rho_qwen05b_90k \
  selection_artifact=/content/drive/MyDrive/diploma/selections/rho_qwen05b_90000/selection_manifest.json \
  adapter_path=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k/adapter \
  output_dir=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k
```

Outputs:

```text
outputs/<experiment>/evaluation/
├── common_eval_metrics.json
├── train_audit_indices.npy
├── config.resolved.json
└── environment.json
```

Common loss is token-weighted assistant-only cross-entropy on the shared holdout. Train audit is a deterministic 1,000-row sample of each method's selected training set. Because methods select radically different answer lengths and difficulty distributions, the train-audit gap is diagnostic, not a clean standalone estimate of generalization.

### 6.4 Public ruMMLU

```bash
!python scripts/benchmark.py \
  experiment_name=rho_qwen05b_90k \
  adapter_path=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k/adapter \
  benchmark_output_dir=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k/benchmarks/rummlu
```

This evaluates 9,748 examples over 57 subjects with five-shot prompts. It uses `gametwix/rummlu`, not the closed MERA leaderboard test.

Outputs:

```text
benchmarks/rummlu/
├── benchmark_run_manifest.json
├── rummlu_metrics.json
└── subjects/<subject>.json
```

Subject files contain every prediction and candidate log-likelihood. They are required for paired analysis.

### 6.5 Public MERA Core

```bash
!python scripts/benchmark.py \
  benchmark=mera_core \
  experiment_name=rho_qwen05b_90k \
  adapter_path=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k/adapter \
  benchmark_output_dir=/content/drive/MyDrive/diploma/outputs/rho_qwen05b_90k/benchmarks/mera_core
```

This evaluates 2,967 public labeled examples from PARus, RCB, RWSD, ruOpenBookQA, and ruWorldTree. It is not a closed MERA leaderboard score.

Outputs:

```text
benchmarks/mera_core/
├── benchmark_run_manifest.json
├── mera_core_metrics.json
└── tasks/<task>.json
```

Use both metrics:

- `accuracy`: micro average, dominated by the large ruOpenBookQA task.
- `macro_task_accuracy`: equal average over the five tasks.

Both benchmark modes checkpoint predictions atomically every 100 examples and resume from existing compatible task/subject files.

### 6.6 Paired statistical analysis

Run paired analysis only after both benchmark directories are complete. Example, random versus entropy on MERA:

```bash
!python scripts/compare_benchmarks.py \
  --baseline-dir /content/drive/MyDrive/diploma/outputs/baseline_random_qwen05b_90k/benchmarks/mera_core \
  --candidate-dir /content/drive/MyDrive/diploma/outputs/entropy_qwen05b_90k/benchmarks/mera_core \
  --output-dir /content/drive/MyDrive/diploma/comparisons/qwen05b_random_vs_entropy/mera_core \
  --baseline-name random \
  --candidate-name entropy
```

Repeat with `benchmarks/rummlu`. Outputs:

- `paired_analysis.json`: accuracy delta, paired bootstrap 95% CI, exact McNemar p-value, agreement, and per-group results.
- `disagreements.jsonl`: all examples where only one model was correct.

Do not infer significance from aggregate accuracy alone. A difference is supported when the paired CI excludes zero and the McNemar result is consistent with that direction.

## 7. Smoke tests before full runs

A new or changed model-based selector must pass a GPU smoke test before a full run. RHO example:

```bash
!python scripts/select_data.py \
  selection=rho \
  selection.pool_size=256 \
  selection.subsample_size=64 \
  selection.rho_holdout_size=128 \
  selection.rho_il_max_steps=5 \
  selection.rho_il_eval_steps=5 \
  selection.rho_il_save_steps=5 \
  selection.rho_il_warmup_steps=1 \
  selection.batch_size=2 \
  selection_output_dir=/content/drive/MyDrive/diploma/selections/smoke_rho_qwen05b_64
```

For IFD or entropy, use `pool_size=256`, `subsample_size=64`, `batch_size=2`, and a fresh smoke output directory. For target training, use a smoke selection artifact plus `max_steps=5 train.logging_steps=1`.

A smoke run proves that data loading, model loading, masks, caches, artifacts, and W&B wiring work. It does not estimate method quality.

## 8. W&B contract

W&B job types separate the stages:

| Job type | Key logs |
|---|---|
| `selection` | score distribution, selected-score distribution, NaN count, method metadata |
| `train` | loss, assistant-token entropy, grad norm, learning rate, epoch, global step, runtime, FLOPs |
| `evaluation` | common loss/PPL, train-audit loss/PPL, token counts, gap |
| `benchmark` | aggregate and per-subject/task metrics, runtime, truncation count |

Selection artifacts, final adapters, evaluation JSON, and benchmark prediction files are uploaded to W&B. Drive remains the operational store for resumable checkpoints and caches.

Do not compare W&B's last logged `train/loss` with final common-eval loss as if they were the same statistic. Training uses packed batches and logs interval values; common evaluation is a separate token-weighted assistant-only pass over fixed examples.

## 9. Comparability checklist

Before comparing two results, verify:

1. Same base model and target training config.
2. Same `training_protocol_sha256`.
3. Same dataset name, resolved revision, and common holdout fingerprint.
4. Different `selection_indices_sha256` only because the selection differs.
5. Same benchmark dataset resolved revision and benchmark scope.
6. Same number of benchmark examples and same truncation policy.
7. Paired prediction files align by group and index.
8. Report optimizer steps, assistant-token counts, FLOPs, and runtime alongside the fixed 90k row budget.

The current experiments are fixed-document-budget comparisons, not fixed-token or fixed-compute comparisons. Selected answer lengths differ substantially, so methods naturally produce different packed steps and FLOPs.

## 10. Current results as of 2026-08-15

All rows below share training protocol SHA `d83f47354a5e861317ecab355b90dbd53da5add6cf5498d312d7756c995ea3da` and common holdout fingerprint `ff5a44a1c8705c4f`.

Known canonical selection lineage:

| Method | `selection_indices_sha256` |
|---|---|
| Random | `2113116881adb139d6f3cb51ec12ebc679793c400dcf6bb57cd537c71dfba719` |
| IFD | `ab2f12e4b37a074ca994bed7f058414814b896240ac4861a2691b84433496d50` |
| Entropy | `5ea7605893113df7e2a3c65b2e65ffea1a667be92ca54b0a536d91e8f03bcb65` |
| RHO | `48b8ba8fe05752360136dc1c94d09d22af0a985f13afd2ae58003d7a1bbde1eb` |

| Method | Common loss | Common PPL | ruMMLU accuracy | MERA micro | MERA macro |
|---|---:|---:|---:|---:|---:|
| Random | 1.1920 | 3.2938 | 29.75% | 34.01% | 36.40% |
| IFD | **1.1679** | **3.2153** | 25.87% | 30.50% | 36.98% |
| Entropy | 1.2543 | 3.5053 | **30.43%** | **38.59%** | **40.77%** |
| RHO | 1.3131 | 3.7176 | 30.11% | 33.81% | 38.02% |

Interpretation:

- IFD improves in-domain common likelihood but significantly harms external benchmarks versus random.
- Entropy is the current external-benchmark leader despite worse common likelihood.
- RHO is approximately random on external aggregates and selects very short responses; it does not beat entropy.
- Common assistant-only loss is not a sufficient model-selection criterion for this project.

Approximate assistant tokens per 1,000-row train audit:

| Method | Tokens | Tokens/row |
|---|---:|---:|
| Random | 215,080 | 218 |
| IFD | 364,532 | 365 |
| Entropy | 128,350 | 128 |
| RHO | 52,408 | 52 |

This length shift must be discussed when interpreting compute and generalization gaps.

## 11. Failure and recovery guide

### Wrong Colab directory

Symptoms: `not a git repository`, `file:///content does not appear to be a Python project`.

Recovery:

```bash
%cd /content/diploma-public
!git status
!pip install -e ".[experiments]"
```

### No GPU or no bf16

`nvidia-smi: command not found` means the notebook has no GPU runtime. Select a GPU runtime and restart setup. The experiment intentionally does not fall back to fp16.

### Existing output belongs to another protocol

Do not bypass the error or edit the manifest. Use a new output directory. If the directory was only a failed smoke attempt and deletion is desired, get explicit user approval first.

### Interrupted selection

Repeat the exact command. Score arrays contain NaN for unfinished rows and continue from the remaining work. RHO resumes IL training first, then base/IL scoring.

### Interrupted target training

Repeat the exact command with the same selection artifact and output directory. `resume_from_checkpoint=auto` finds the latest checkpoint.

### Interrupted benchmark

Repeat the exact command. Completed predictions in compatible subject/task files are retained.

### Unsloth import-order lint warning

Unsloth must remain before Transformers/TRL even if a generic import sorter objects. Use the established local `# noqa: I001`; do not reorder it below Transformers.

## 12. Extending the experiment

The next planned method is the quality classifier. A proper port must:

1. Add `configs/selection/quality.yaml`.
2. Put reusable pure logic under `diploma_sft/` with CPU tests.
3. Add a `quality` branch to `scripts/select_data.py` without changing `train.py`.
4. Version labels, labeling model/protocol, scorer model commit, classifier split, score cache, and selected indices.
5. Keep the common holdout completely inaccessible. Version the labeled scorer-training subset separately; either exclude those rows from candidate ranking or explicitly measure and document in-sample scoring bias.
6. Avoid requiring an API key when a complete immutable label artifact already exists; otherwise document secret handling without storing keys.
7. Add a 256-row GPU smoke command and pass it before the 200k selection.

After the classifier, the most informative follow-up is entropy plus diversity or a length/compute-controlled entropy ablation. Repeated training seeds are required before making a strong claim that entropy is robustly superior.

## 13. Local verification before pushing code

```bash
python3 -m ruff check notebooks/ifd_select.py tests/
python3 -m ruff check scripts/ diploma_sft/
python3 -m compileall -q notebooks/ifd_select.py diploma_sft scripts
python3 -m pytest tests/ -q
python3 scripts/select_data.py selection=rho --cfg job --resolve
python3 -m build
```

GPU-specific behavior cannot be fully verified on the local CPU-only workstation. State that explicitly, then run the smallest relevant Colab smoke before authorizing a full experiment.

"""Evaluate a trained adapter on the shared no-leak common holdout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import hydra
    from hydra.utils import to_absolute_path
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing experiment dependencies. Install them with: "
        'python -m pip install -e ".[experiments]"'
    ) from exc

from diploma_sft.artifacts import load_selection_artifact, validate_adapter_lineage  # noqa: E402
from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import load_common_evaluation_dataset  # noqa: E402
from diploma_sft.evaluation import compute_assistant_only_perplexity  # noqa: E402
from diploma_sft.runtime import environment_snapshot, require_bf16_cuda  # noqa: E402
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.selection_artifact:
        raise ValueError("selection_artifact is required")
    if not cfg.adapter_path:
        raise ValueError("adapter_path is required")

    manifest_path = Path(to_absolute_path(str(cfg.selection_artifact)))
    manifest, _ = load_selection_artifact(manifest_path)
    adapter_path = Path(to_absolute_path(str(cfg.adapter_path)))
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    training_run_manifest = validate_adapter_lineage(
        adapter_path,
        selection_indices_sha256=manifest["indices"]["sha256"],
    )
    output_dir = Path(to_absolute_path(str(cfg.evaluation_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    _write_json(output_dir / "environment.json", environment_snapshot(REPO_ROOT))

    require_bf16_cuda()
    # Unsloth must patch Transformers before model/tokenizer loading.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    FastLanguageModel.for_inference(model)

    common_eval, dataset_metadata = load_common_evaluation_dataset(
        manifest,
        tokenizer=tokenizer,
        sample_limit=cfg.dataset.common_eval_samples,
    )
    started = time.perf_counter()
    metrics = compute_assistant_only_perplexity(
        model,
        tokenizer,
        common_eval,
        max_length=cfg.model.max_seq_len,
    )
    metrics.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "adapter_path": str(adapter_path),
            "selection_method": manifest["selection"]["method"],
            "selection_indices_sha256": manifest["indices"]["sha256"],
            "training_protocol_sha256": training_run_manifest[
                "training_protocol_sha256"
            ],
            "dataset": dataset_metadata,
        }
    )
    metrics_path = output_dir / "common_eval_metrics.json"
    _write_json(metrics_path, metrics)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="evaluation",
        extra_tags=[manifest["selection"]["method"]],
    )
    if run is not None:
        run.log(
            {
                "common_eval/assistant_only_loss": metrics["assistant_only_loss"],
                "common_eval/assistant_only_perplexity": metrics[
                    "assistant_only_perplexity"
                ],
                "common_eval/assistant_tokens": metrics["assistant_tokens"],
                "common_eval/evaluated_examples": metrics["evaluated_examples"],
                "common_eval/skipped_examples": metrics["skipped_examples"],
            }
        )
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-common-eval",
        artifact_type="evaluation",
        paths=[str(metrics_path)],
        metadata={
            "selection_method": manifest["selection"]["method"],
            "selection_indices_sha256": manifest["indices"]["sha256"],
            "training_protocol_sha256": training_run_manifest[
                "training_protocol_sha256"
            ],
        },
    )
    if run is not None:
        run.finish()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Recover verified stage boundaries without restoring unsaved optimizer state."""

import json
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, file_digest, tree_identity
from simple_scibuddy.coevolve.protocol import read_request, validate_result


def completed_boundary(source, config, identities):
    """Recover a committed stage boundary; never claim to restore unsaved optimizer updates."""
    source = Path(source).resolve()
    if not source.is_relative_to(ROOT / "runs"):
        raise ValueError("Continuation source must belong to this repository")
    previous = json.loads((source / "config.json").read_text())
    if any(previous[k] != config[k] for k in ("mode", "model", "harness_evolve")):
        raise ValueError("Continuation must preserve model and harness experiment settings")
    pinned = json.loads((source / "identities.json").read_text())
    if any(pinned[k] != identities[k] for k in ("dataset", "runtime", "verifier")):
        raise ValueError("Continuation dataset/runtime/verifier identity changed")
    results = sorted((source / "shared/rounds").glob("r[0-9][0-9][0-9][0-9]/result.json"))
    if not results:
        raise ValueError("No completed RL stage is available for continuation")
    directory = results[-1].parent
    result = validate_result(directory)
    number = int(directory.name[1:])
    if number > config["coevolve"].get("max_rounds", 1):
        raise ValueError("Continuation is beyond the configured round budget")
    phase = f"h{number:04d}"
    summary = json.loads((source / "harness_evolve" / phase / "summary.json").read_text())
    harness = Path(summary["selected_harness"])
    if file_digest(harness) != result["trained_with_harness_sha256"]:
        raise ValueError("Continuation harness differs from the completed training stage")
    history = json.loads((source / "harness_evolve-history.json").read_text())
    if result.get("status") == "failed":
        request = read_request(directory)
        status = json.loads((source / "status.json").read_text())
        if status.get("status") != "failed":
            raise ValueError("Cannot restart an unfinished/live source experiment")
        if (
            summary["steps"] != config["harness_evolve"]["steps_per_phase"]
            or summary["selected_harness_sha256"] != request["harness_sha256"]
            or summary["selected"] != request["eval"]["metrics"]
            or summary["initial"]["correct"] != request["reference_correct"]
            or str(Path(summary["model"]).resolve()) != request["base_model_path"]
        ):
            raise ValueError("Completed harness summary does not match the published handoff")
        if number == 1:
            if tree_identity(request["base_model_path"]) != pinned["model"]:
                raise ValueError("Original M0 identity changed before RL restart")
        else:
            previous_round = validate_result(directory.parent / f"r{number - 1:04d}")
            if (
                not previous_round
                or previous_round.get("status") != "completed"
                or previous_round["model_path"] != request["base_model_path"]
            ):
                raise ValueError("RL restart requires the preceding completed model checkpoint")
        return {
            "source": str(source),
            "boundary": "completed_harness",
            "rounds": number - 1,
            "model": request["base_model_path"],
            "harness": str(harness),
            "train_cursor": summary["train_cursor"],
            "history": [h for h in history if h.get("phase", "") <= phase],
            "phase_summary": summary,
            "phase_summary_sha256": file_digest(source / "harness_evolve" / phase / "summary.json"),
            "request_sha256": file_digest(directory / "request.json"),
            "result_sha256": file_digest(directory / "result.json"),
            "optimizer_recovery": "Unsaved updates are not restored; restart this RL stage from its pinned base model.",
        }
    return {
        "source": str(source),
        "rounds": number,
        "model": result["model_path"],
        "harness": str(harness),
        "train_cursor": summary["train_cursor"],
        "history": [h for h in history if h.get("phase", "") <= phase],
        "result_sha256": file_digest(directory / "result.json"),
    }

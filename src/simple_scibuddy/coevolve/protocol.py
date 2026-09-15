"""ScienceBuddy co-evolution protocol v1; standard-library-only RL integration CLI."""

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if json.loads(path.read_text()) != value:
                    raise ValueError(f"Conflicting immutable publication: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_request(round_dir):
    directory = Path(round_dir).resolve()
    request = json.loads((directory / "request.json").read_text())
    if request["protocol_version"] != 1 or request["round_id"] != directory.name:
        raise ValueError("Unsupported protocol or wrong round_id")
    file = directory / request["harness_file"]
    if file.resolve().parent != directory or sha256(file) != request["harness_sha256"]:
        raise ValueError("Harness file/hash mismatch")
    return request


def publish_request(
    root,
    round_id,
    harness,
    *,
    base_model_id,
    base_model_path,
    harness_id,
    evaluation,
    reference_correct,
    contract,
    source_run,
):
    if not round_id.startswith("r") or not round_id[1:].isdigit():
        raise ValueError("Invalid round_id")
    shared = Path(root).resolve()
    shared.mkdir(parents=True, exist_ok=True)
    rounds = shared / "rounds"
    rounds.mkdir(exist_ok=True)
    if shared.stat().st_mode & stat.S_IWGRP:
        rounds.chmod(rounds.stat().st_mode | stat.S_IWGRP | stat.S_ISGID)
    directory = rounds / round_id
    directory.mkdir(exist_ok=True)
    if rounds.stat().st_mode & stat.S_IWGRP:
        directory.chmod(directory.stat().st_mode | stat.S_IWGRP | stat.S_ISGID)
    request = {
        "protocol_version": 1,
        "round_id": round_id,
        "base_model_id": base_model_id,
        "base_model_path": str(Path(base_model_path).resolve()),
        "harness_id": harness_id,
        "harness_file": "harness.py",
        "harness_sha256": sha256(harness),
        "eval": evaluation,
        "reference_correct": reference_correct,
        "runtime_contract": contract,
        "source_run": str(Path(source_run).resolve()),
    }
    if (directory / "request.json").exists():
        if read_request(directory) != request:
            raise ValueError("Existing round belongs to a different request")
        return directory
    destination = directory / "harness.py"
    if destination.exists() and sha256(destination) != request["harness_sha256"]:
        raise ValueError("Incomplete round has a different harness")
    if not destination.exists():
        tmp = directory / (".harness." + uuid.uuid4().hex + ".tmp")
        tmp.write_bytes(Path(harness).read_bytes())
        try:
            os.link(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
    atomic_json(directory / "request.json", request, immutable=True)  # The readiness signal, written LAST.
    return directory


def claim(round_dir, worker_id):
    directory = Path(round_dir)
    request = read_request(directory)
    body = {
        "protocol_version": 1,
        "round_id": request["round_id"],
        "request_sha256": sha256(directory / "request.json"),
        "worker_id": worker_id,
    }
    atomic_json(directory / "accepted.json", body, immutable=True)
    return body


def checkpoint_manifest(model_path):
    root = Path(model_path).resolve()
    if not (root / "config.json").is_file():
        raise ValueError("Full checkpoint requires config.json")
    if not (root / "tokenizer_config.json").is_file():
        raise ValueError("Full checkpoint requires tokenizer_config.json")
    if not any((root / n).is_file() for n in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Full checkpoint requires tokenizer files")
    indexes = list(root.glob("*.index.json"))
    weight_names = set()
    for index in indexes:
        content = json.loads(index.read_text())
        weight_names.update(content.get("weight_map", {}).values())
    if not weight_names:
        weight_names = {p.name for p in root.glob("*.safetensors")} | {
            p.name for p in root.glob("pytorch_model*.bin")
        }
    if not weight_names:
        raise ValueError("No full model weights found; adapters alone are not supported in v1")
    files = {
        p.name
        for p in root.iterdir()
        if p.is_file() and (p.suffix in {".json", ".model", ".txt", ".tiktoken"} or p.name in weight_names)
    } | weight_names
    files.discard("sciencebuddy-model-manifest.json")
    rows = []
    for name in sorted(files):
        path = root / name
        if path.resolve().parent != root or not path.is_file():
            raise ValueError("Missing or nonlocal checkpoint file: " + name)
        rows.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"format": "sciencebuddy-checkpoint-v1", "files": rows}
    path = root / "sciencebuddy-model-manifest.json"
    atomic_json(path, manifest, immutable=True)
    return path


def publish_result(round_dir, worker_id, model_id=None, model_path=None, *, failed=None, load_checked=False):
    directory = Path(round_dir).resolve()
    request = read_request(directory)
    accepted = json.loads((directory / "accepted.json").read_text())
    request_hash = sha256(directory / "request.json")
    if accepted["worker_id"] != worker_id or accepted["request_sha256"] != request_hash:
        raise ValueError("This worker does not own this request")
    body = {
        "protocol_version": 1,
        "round_id": request["round_id"],
        "request_sha256": request_hash,
        "base_model_id": request["base_model_id"],
        "trained_with_harness_sha256": request["harness_sha256"],
        "worker_id": worker_id,
        "status": "failed" if failed else "completed",
    }
    if failed:
        body["error"] = failed
    else:
        if not load_checked:
            raise ValueError("Confirm a successful local checkpoint load before publishing")
        body["producer_load_checked"] = True
        if (
            not model_id
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id)
            or model_id == request["base_model_id"]
        ):
            raise ValueError("Provide a new unique model_id")
        path = Path(model_path).resolve()
        if path == Path(request["base_model_path"]).resolve():
            raise ValueError("Save M1 in a NEW directory; never overwrite M0")
        manifest = checkpoint_manifest(path)
        body.update(
            model_id=model_id,
            artifact_type="full_checkpoint",
            model_path=str(path),
            manifest_file=manifest.name,
            manifest_sha256=sha256(manifest),
        )
    atomic_json(directory / "result.json", body, immutable=True)
    return body


def validate_result(round_dir):
    directory = Path(round_dir).resolve()
    request = read_request(directory)
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    for key, value in {
        "protocol_version": 1,
        "round_id": request["round_id"],
        "request_sha256": sha256(directory / "request.json"),
        "base_model_id": request["base_model_id"],
        "trained_with_harness_sha256": request["harness_sha256"],
    }.items():
        if result.get(key) != value:
            raise ValueError("Result mismatch: " + key)
    accepted = json.loads((directory / "accepted.json").read_text())
    if (
        result.get("worker_id") != accepted["worker_id"]
        or accepted["request_sha256"] != result["request_sha256"]
    ):
        raise ValueError("Result worker/request does not match the claim")
    if result["status"] == "failed":
        return result
    if not result.get("producer_load_checked"):
        raise ValueError("Producer has not confirmed checkpoint load")
    if result["status"] != "completed" or result.get("artifact_type") != "full_checkpoint":
        raise ValueError("Unsupported result status/artifact type")
    root = Path(result["model_path"]).resolve()
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", result.get("model_id", ""))
        or result["model_id"] == request["base_model_id"]
        or root == Path(request["base_model_path"]).resolve()
    ):
        raise ValueError("M1 must have a new identity and directory")
    manifest = root / result["manifest_file"]
    if manifest.resolve().parent != root or sha256(manifest) != result["manifest_sha256"]:
        raise ValueError("Model manifest mismatch")
    data = json.loads(manifest.read_text())
    if data.get("format") != "sciencebuddy-checkpoint-v1":
        raise ValueError("Unknown model manifest")
    names = set()
    for row in data["files"]:
        file = root / row["path"]
        if row["path"] in names or file.resolve().parent != root or not file.is_file():
            raise ValueError("Invalid model file entry")
        names.add(row["path"])
        if file.stat().st_size != row["size"] or sha256(file) != row["sha256"]:
            raise ValueError("Model file mismatch: " + row["path"])
    expected = json.loads(checkpoint_manifest(root).read_text())
    if expected != data:
        raise ValueError("Incomplete model manifest")
    return result


def retry_failed_round(round_dir):
    directory = Path(round_dir).resolve()
    request = read_request(directory)
    result = validate_result(directory)
    if result is None or result["status"] != "failed":
        raise ValueError("Only an explicitly failed round can be retried")
    state_path = Path(request["source_run"]) / "coevolve-state.json"
    state = json.loads(state_path.read_text())
    if state["awaiting"] != str(directory) or state.get("loading"):
        raise ValueError("Coordinator is not waiting on this failed round")
    next_id = f"r{state['round_number'] + 1:04d}"
    new = publish_request(
        directory.parent.parent,
        next_id,
        directory / "harness.py",
        base_model_id=request["base_model_id"],
        base_model_path=request["base_model_path"],
        harness_id=request["harness_id"],
        evaluation=request["eval"],
        reference_correct=request["reference_correct"],
        contract=request["runtime_contract"],
        source_run=request["source_run"],
    )
    state.update(awaiting=str(new), round_number=state["round_number"] + 1)
    atomic_json(state_path, state)
    return {"retry_round": str(new)}


def main():
    """Explicit inspection/recovery only; the experiment owns normal handoffs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "retry"])
    parser.add_argument("--round", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "retry":
        result = retry_failed_round(args.round)
    else:
        result = validate_result(args.round)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

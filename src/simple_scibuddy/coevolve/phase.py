"""Execute one harness evolution stage with a fixed model and validation panel."""

import asyncio
import os
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from simple_scibuddy.artifacts import digest, file_digest, write_json
from simple_scibuddy.coevolve.capabilities import dataset_capabilities
from simple_scibuddy.coevolve.evidence import preflight_passed, public_evidence
from simple_scibuddy.coevolve.feedback import review_submission
from simple_scibuddy.coevolve.improver import Improver
from simple_scibuddy.coevolve.program import propose
from simple_scibuddy.coevolve.selection import select_candidates
from simple_scibuddy.harness.broker import run_episode
from simple_scibuddy.harness.scientific import SYSTEM
from simple_scibuddy.inference.client import RecordingPolicy
from simple_scibuddy.inference.server import CompletionClient, serve, serve_pool


def summarize(episodes):
    n = len(episodes)
    return {
        "accuracy": sum(e["reward"] == 1 for e in episodes) / n,
        "correct": sum(e["reward"] == 1 for e in episodes),
        "tasks": n,
        "not_submitted_rate": sum(not e["submissions"] for e in episodes) / n,
        "context_limit_rate": sum(e["stop_reason"] == "context_budget" for e in episodes) / n,
        "response_limit_rate": sum(e["stop_reason"] == "output_truncated" for e in episodes) / n,
        "call_limit_rate": sum(
            e["stop_reason"] in {"tool_call_budget", "model_call_budget"} for e in episodes
        )
        / n,
        "mean_model_calls": sum(len(e["calls"]) for e in episodes) / n,
        "mean_tool_calls": sum(len(e["tool_calls"]) for e in episodes) / n,
        "tool_error_rate": sum(bool(t["observation"].get("error")) for e in episodes for t in e["tool_calls"])
        / max(1, sum(len(e["tool_calls"]) for e in episodes)),
        "mean_response_tokens": sum(len(c["completion_ids"]) for e in episodes for c in e["calls"]) / n,
        "feedback_count": sum(len(e["user_feedback"]) for e in episodes),
        "feedback_service_error_count": sum(
            bool(f.get(k, {}).get("service_error"))
            for e in episodes
            for f in e["user_feedback"]
            for k in ("user_generation", "prm_generation")
        ),
        "feedback_template_fallback_count": sum(
            bool(f.get("template_fallback")) for e in episodes for f in e["user_feedback"]
        ),
        "feedback_invalid_count": sum(
            not f.get("valid", False) for e in episodes for f in e["user_feedback"]
        ),
    }


async def batch(dataset, rows, harness, url, tokenizer, cfg, folder, reviewer=None, *, limit=None):
    clients = [CompletionClient(u) for u in ([url] if isinstance(url, str) else url)]
    if limit is None:
        limit = asyncio.Semaphore(cfg["workers"])

    async def one(index, row):
        async with limit:
            client = clients[index % len(clients)]
            seed = int(digest([cfg["seed"], row["id"]])[:8], 16) % (2**31)
            policy = RecordingPolicy(
                client,
                tokenizer,
                {"temperature": 1.0 if reviewer else 0.0, "top_p": 1.0, "seed": seed},
                client.model_name,
                row["id"],
                cfg["context_tokens"],
                cfg["max_tokens"],
            )
            episode = await run_episode(
                dataset.load(row["id"]),
                harness,
                policy,
                folder / row["id"],
                SYSTEM,
                actions=cfg["actions"],
                seconds=cfg["seconds"],
                tool_seconds=cfg["tool_seconds"],
                runtime_image=cfg["runtime_image"],
                baked_lake=cfg.get("baked_lake", False),
                tool_response_tokens=cfg["tool_response_tokens"],
                tool_history_tokens=cfg["tool_history_tokens"],
                feedback=partial(review_submission, dataset.load(row["id"]), reviewer) if reviewer else None,
                user_turns=cfg.get("user_turns", 1) if reviewer else 0,
            )
            episode.update(
                model_path=cfg["model"],
                seed=seed,
                public_task=dataset.load(row["id"]).public,
                harness_sha256=file_digest(harness),
            )
            write_json(folder / row["id"] / "episode.json", episode)
            print(
                f"Harness rollout: {folder.name}/{row['id']}, reward={episode['reward']}, "
                f"stop={episode['stop_reason']}",
                flush=True,
            )
            return episode

    try:
        async with asyncio.TaskGroup() as group:
            jobs = [group.create_task(one(i, row)) for i, row in enumerate(rows)]
        return [job.result() for job in jobs]
    finally:
        await asyncio.gather(*(client.close() for client in clients))


def harness_evolve(
    root, phase_id, model, harness, train, cursor, dataset, cfg, options, tracker, round_number, history
):
    """Configured evolution steps under one frozen model; keep services alive for the phase."""
    from transformers import AutoTokenizer

    phase = root / "harness_evolve" / phase_id
    phase.mkdir(parents=True)
    prefix = f"harness_evolve/{phase_id}"
    tracker.define_metric(prefix + "/*", step_metric=prefix + "/step")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    selection_rows = dataset.tasks("val")
    if not selection_rows or len(selection_rows) != options.get("selection_tasks", 90):
        raise ValueError("selection_tasks must equal the explicit validation split size")
    write_json(
        phase / "selection-tasks.json",
        {"task_ids": [r["id"] for r in selection_rows], "source": "val", "excluded_from_interactions": True},
    )
    initial_harness = harness
    cfg = dict(cfg, model=model)

    def log(step, values):
        record = {
            prefix + "/step": step,
            prefix + "/round": round_number,
            **{prefix + "/" + k: v for k, v in values.items()},
        }
        tracker.log(record)
        return record

    def status(step):
        write_json(
            root / "status.json",
            {
                "status": "harness_evolve",
                "phase": phase_id,
                "step": step,
                "round": round_number,
                "train_cursor": cursor,
            },
        )

    status(0)
    with ExitStack() as services:
        icfg = dict(options["improver"])
        managed = not icfg.get("base_url")
        same_model = managed and Path(icfg["model"]).resolve() == Path(model).resolve()
        improver_tokenizer = (
            (tokenizer if same_model else AutoTokenizer.from_pretrained(icfg["model"], local_files_only=True))
            if managed
            else None
        )
        url = services.enter_context(
            serve_pool(
                model,
                devices,
                phase / "solver",
                max(cfg["context_tokens"], 32768),
                share_last=managed and not same_model,
            )
        )
        if not icfg.get("base_url"):
            icfg["base_url"] = (
                url[-1]
                if same_model
                else services.enter_context(serve(icfg["model"], devices[-1], phase / "improver", memory=0.4))
            )
            icfg["model"] = "sciencebuddy"
        reviewer = Improver(
            icfg,
            tokenizer=improver_tokenizer,
            context_tokens=(max(cfg["context_tokens"], 32768) if same_model else 32768) if managed else None,
        )
        preflight_rows = [next((r for r in train if r["id"] == options.get("preflight_task_id")), train[0])]
        if options.get("preflight_task_id") and preflight_rows[0]["id"] != options["preflight_task_id"]:
            raise ValueError("Preflight task must belong to training split")
        environment = dataset_capabilities(dataset)
        write_json(phase / "public-environment.json", environment)

        def preflight(path, folder):
            return preflight_passed(
                asyncio.run(batch(dataset, preflight_rows, path, url, tokenizer, cfg, folder))[0]
            )

        if not preflight(harness, phase / "initial-preflight"):
            raise RuntimeError("Inherited harness failed execution preflight")
        initial = selected = summarize(
            asyncio.run(
                batch(dataset, dataset.tasks("test"), harness, url, tokenizer, cfg, phase / "baseline")
            )
        )
        log(0, {"evaluation/" + k: v for k, v in initial.items()})
        pending = []
        for step in range(1, options["steps_per_phase"] + 1):
            folder = phase / f"step-{step:04d}"
            folder.mkdir()
            status(step)
            step_started = time.monotonic()
            count = options.get("feedback_every", 8)
            rows = [train[(cursor + i) % len(train)] for i in range(count)]
            cursor += count
            episodes = asyncio.run(
                batch(dataset, rows, harness, url, tokenizer, cfg, folder / "interaction", reviewer)
            )
            interaction_seconds = time.monotonic() - step_started
            current = [public_evidence(e) for e in episodes]
            pending.extend(current)
            failures = [r for r in pending[: -len(current)] if r["feedback"] or r["diagnostics"]][-12:]
            controls = [r for r in pending[: -len(current)] if r.get("acceptance") and not r["diagnostics"]][
                -4:
            ]
            parent = harness

            def proposal(path, index):
                return propose(
                    reviewer,
                    parent,
                    failures + controls + current,
                    history,
                    path,
                    profile={
                        **{k: cfg[k] for k in ("model", "actions", "context_tokens", "max_tokens")},
                        "candidate_index": index,
                        "candidate_count": options.get("candidates", 3),
                        "instruction": "Propose an independent hypothesis from this same parent; explore a useful alternative.",
                    },
                    environment=environment,
                )

            validation_limit = None

            async def evaluate(path, destination):
                nonlocal validation_limit
                if validation_limit is None:
                    validation_limit = asyncio.Semaphore(cfg["workers"])
                return await batch(
                    dataset, selection_rows, path, url, tokenizer, cfg, destination, limit=validation_limit
                )

            selection_started = time.monotonic()
            result = select_candidates(
                parent,
                folder,
                options.get("candidates", 3),
                proposal,
                preflight,
                evaluate,
                [r["id"] for r in selection_rows],
            )
            applied = result["status"] == "applied"
            harness = Path(result["selected"])
            if applied:
                pending.clear()
            result.update(phase=phase_id, step=step, parent=str(parent), train_cursor=cursor)
            # Only aggregate validation feedback and proposed text enter future prompts.
            history.append(
                {k: result[k] for k in ("status", "reason", "operations", "hypothesis", "phase", "step")}
                | {"selection_validation": result["validation"]}
            )
            write_json(folder / "update.json", result)
            write_json(root / "harness_evolve-history.json", history)
            values = {
                "timing/interaction_seconds": interaction_seconds,
                "timing/candidate_search_seconds": time.monotonic() - selection_started,
                "update_valid": int(any(c["executable"] for c in result["candidates"])),
                "update_applied": int(applied),
                "selected_candidate": result["selected_candidate"],
                **{"validation/" + k: v for k, v in result["validation"].items()},
                **{"interaction/" + k: v for k, v in summarize(episodes).items()},
            }
            for c in result["candidates"]:
                values[f"candidates/c{c['id']:02d}/executable"] = int(c["executable"])
                if c["validation"] is not None:
                    values.update({f"candidates/c{c['id']:02d}/" + k: v for k, v in c["validation"].items()})
            if step == options["steps_per_phase"]:
                selected = summarize(
                    asyncio.run(
                        batch(
                            dataset,
                            dataset.tasks("test"),
                            harness,
                            url,
                            tokenizer,
                            cfg,
                            folder / "evaluation",
                        )
                    )
                )
                values.update({"evaluation/" + k: v for k, v in selected.items()})
            values["timing/step_seconds"] = time.monotonic() - step_started
            write_json(folder / "metrics.json", log(step, values))
    summary = {
        "phase": phase_id,
        "model": model,
        "model_id": f"M{round_number}",
        "initial_harness": str(initial_harness),
        "selected_harness": str(harness),
        "selected_harness_sha256": file_digest(harness),
        "initial": initial,
        "selected": selected,
        "steps": options["steps_per_phase"],
        "train_cursor": cursor,
        "improved": file_digest(harness) != file_digest(initial_harness)
        and selected["correct"] > initial["correct"],
    }
    write_json(phase / "summary.json", summary)
    return harness, cursor, summary

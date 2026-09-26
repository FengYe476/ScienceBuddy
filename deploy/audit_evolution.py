#!/usr/bin/env python3
"""Examine every accepted harness edit: what changed, where, and whether task content leaked in.

    python3 deploy/audit_evolution.py runs/h-only-v3 --release data/releases/id715-val30-test90-v1

Three sections, one per question.

  [1] What each accepted step changed, separated into the two things an edit can touch:
      module-level string constants (the text the solver reads) and the body of run()
      (the control flow the harness itself executes). The paper's Section S1.2 restricts
      its case study to instruction and skill text and freezes the execution loop; the
      repository does not enforce that, so the split is worth measuring.

  [2] Whether task-specific content was baked in. Every distinctive literal in the harness
      is cross-referenced against the actual task payloads:
        * a reference answer of any task          -- outright leakage
        * a task ID                               -- outright leakage
        * text found only in test tasks           -- alarming; the improver never sees test
        * text found in train tasks               -- expected, since train is its evidence,
                                                     but worth reading: a column name
                                                     generalises, a specific gene set does not
      Only identifier-like tokens are considered (>= 4 characters, carrying an underscore,
      a digit or an interior capital), because ordinary prose matches everything.

  [3] How many distinct places each step touched, counted as unified-diff hunks rather than
      lines, so that rewriting one paragraph counts once.

Read-only, pure standard library.
"""

import argparse
import ast
import collections
import difflib
import json
import pathlib
import re
import sys

BAR = "=" * 78
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:\-]{3,}")


def accepted(root):
    """[(name, path, update)] for steps that changed the harness, in order."""
    out = []
    for step in sorted(root.glob("step-*")):
        update = step / "update.json"
        if not update.is_file():
            continue
        try:
            data = json.loads(update.read_text())
        except (OSError, ValueError):
            continue
        if data.get("status") != "applied":
            out.append((step.name, None, data))
            continue
        path = step / f"candidate-{data['selected_candidate']:02d}" / "harness.py"
        out.append((step.name, path if path.is_file() else None, data))
    return out


def parts(source):
    """Module-level string constants and the source of run(), separately."""
    tree = ast.parse(source)
    lines = source.splitlines()
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value.value
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body += lines[node.lineno - 1:node.end_lineno]
    return consts, body


def hunks(before, after):
    """Contiguous changed regions, so one rewritten paragraph counts once."""
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    return [op for op in matcher.get_opcodes() if op[0] != "equal"]


def distinctive(text):
    out = set()
    for token in TOKEN.findall(text):
        if "_" in token or any(c.isdigit() for c in token) or any(c.isupper() for c in token[1:]):
            out.add(token)
    return out


def task_corpus(release):
    """Per split: the distinctive tokens in prompts and options, plus reference answers."""
    manifest = json.loads((release / "manifest.json").read_text())
    tokens = collections.defaultdict(set)
    answers, ids = {}, {}
    for row in manifest["tasks"]:
        base = release / row["id"]
        try:
            public = json.loads((base / "public/task.json").read_text())
            ref = json.loads((base / "evaluator/reference.json").read_text())
        except (OSError, ValueError):
            continue
        blob = public.get("prompt", "") + " " + " ".join(public.get("options") or [])
        tokens[row["split"]] |= distinctive(blob)
        answer = str(ref.get("answer", "")).strip()
        if len(answer) > 2:                       # single letters are not evidence
            answers.setdefault(answer, []).append((row["id"], row["split"]))
        ids[row["id"]] = row["split"]
    return tokens, answers, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--release", required=True)
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--initial", default="src/simple_scibuddy/harness/scientific.py")
    ap.add_argument("--show", type=int, default=6, help="literals to list per category")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    release = pathlib.Path(args.release)
    initial = pathlib.Path(args.initial)
    for path, what in ((root, "run"), (release / "manifest.json", "release"), (initial, "initial harness")):
        if not path.exists():
            sys.exit(f"cannot find {what}: {path}")

    steps = accepted(root)
    chain = [("H0", initial)] + [(name, path) for name, path, _ in steps if path]
    if len(chain) < 2:
        sys.exit("no accepted harness edits in this run")

    # ---------------------------------------------------------------- [1] + [3]
    print(BAR)
    print("[1] what each accepted step changed, and [3] how many places")
    print(BAR)
    print(f"\n  {'step':12}{'lines':>7}{'prompt chars':>14}{'run() lines':>13}"
          f"{'text hunks':>12}{'code hunks':>12}")
    previous = None
    for name, path in chain:
        source = path.read_text()
        consts, body = parts(source)
        text = "\n".join(f"{k}\n{v}" for k, v in sorted(consts.items()))
        if previous is None:
            print(f"  {name:12}{len(source.splitlines()):7}{len(text):14}{len(body):13}"
                  f"{'-':>12}{'-':>12}")
        else:
            th = hunks(previous[0].splitlines(), text.splitlines())
            ch = hunks(previous[1], body)
            print(f"  {name:12}{len(source.splitlines()):7}{len(text):14}{len(body):13}"
                  f"{len(th):12}{len(ch):12}")
        previous = (text, body)

    for (before_name, before_path), (after_name, after_path) in zip(chain, chain[1:]):
        bc, bb = parts(before_path.read_text())
        ac, ab = parts(after_path.read_text())
        print(f"\n  --- {before_name} -> {after_name} ---")
        added_names = sorted(set(ac) - set(bc))
        changed = sorted(k for k in set(ac) & set(bc) if ac[k] != bc[k])
        print(f"      new constants     {added_names or 'none'}")
        print(f"      changed constants {changed or 'none'}")
        for key in changed:
            grew = len(ac[key]) - len(bc[key])
            print(f"          {key}: {len(bc[key])} -> {len(ac[key])} chars ({grew:+d})")
        ch = hunks(bb, ab)
        print(f"      run() edits       {len(ch)} hunk(s)")
        for tag, i1, i2, j1, j2 in ch:
            snippet = next((l.strip() for l in ab[j1:j2] if l.strip()), "(deletion)")
            print(f"          {tag:8} -{i2-i1:2} +{j2-j1:2}   {snippet[:82]}")

    # ---------------------------------------------------------------- [2]
    print("\n" + BAR)
    print("[2] did task-specific content get baked in")
    print(BAR)
    tokens, answers, ids = task_corpus(release)
    train_only = tokens["train"] - tokens["test"] - tokens["val"]
    test_only = tokens["test"] - tokens["train"] - tokens["val"]

    for name, path in chain[1:]:
        source = path.read_text()
        found = distinctive(source)
        print(f"\n  [{name}]  {len(source.splitlines())} lines, {len(found)} distinctive literals")

        leaked_answers = sorted(a for a in answers
                                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", source))
        leaked_ids = sorted(t for t in ids if t in source)
        hit_test = sorted(found & test_only)
        hit_train = sorted(found & train_only)

        lines = source.splitlines()

        def context(needle):
            """Where the literal sits, so a citation in a comment can be told from a
            hardcoded answer. Severity depends entirely on this."""
            out = []
            for i, line in enumerate(lines):
                if needle in line:
                    stripped = line.strip()
                    kind = "comment" if stripped.startswith("#") else "code/text"
                    out.append((i + 1, kind, stripped[:96]))
            return out

        flag = lambda n: "!! " if n else "   "
        print(f"      {flag(leaked_answers)}reference answers verbatim   {len(leaked_answers)}")
        for a in leaked_answers[:args.show]:
            where = ", ".join(f"{tid}({sp})" for tid, sp in answers[a])
            print(f"           {a!r}  is the answer to {where}")
            for ln, kind, snippet in context(a)[:3]:
                print(f"             line {ln} [{kind}] {snippet}")
        print(f"      {flag(leaked_ids)}task IDs                     {len(leaked_ids)}")
        for t_id in leaked_ids[:args.show]:
            print(f"           {t_id}  (split: {ids[t_id]})")
            for ln, kind, snippet in context(t_id)[:2]:
                print(f"             line {ln} [{kind}] {snippet}")
        print(f"      {flag(hit_test)}literals seen only in test   {len(hit_test)}"
              + (f"  {hit_test[:args.show]}" if hit_test else ""))
        print(f"         literals seen only in train  {len(hit_train)}"
              + (f"  {hit_train[:args.show]}" if hit_train else ""))
        for lit in hit_train[:args.show]:
            for ln, kind, snippet in context(lit)[:1]:
                print(f"           line {ln} [{kind}] {snippet}")

    print(f"\n  Corpus: {len(tokens['train'])} distinctive literals in train, "
          f"{len(tokens['test'])} in test, {len(answers)} multi-character reference answers.\n")


if __name__ == "__main__":
    main()

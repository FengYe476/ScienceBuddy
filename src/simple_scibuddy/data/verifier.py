import re


def normalized(text):
    return " ".join(text.split()).casefold()


def grade_answer(task, reference, response):
    """Only the explicitly submitted answer is scored, never arbitrary prose."""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.S | re.I)
    answer = matches[-1].strip() if len(matches) == 1 else ""
    raw_answer = answer
    options = task.get("options", [])
    # A real rollout submitted 'D. Greatly reduced': same choice, not a scientific error.
    labelled = re.fullmatch(r"([A-Za-z])[.)]\s*(.+)", answer, re.S)
    if labelled and options:
        index = ord(labelled[1].upper())-65
        if 0 <= index < len(options) and normalized(labelled[2]).rstrip(".") == normalized(options[index]).rstrip("."):
            answer = labelled[1].upper()
    elif options:
        selected = [i for i, option in enumerate(options) if normalized(option) == normalized(answer)]
        if len(selected) == 1:
            answer = chr(65+selected[0])
    expected = reference["answer"]
    passed = (answer.strip() == expected.strip() if task.get("subtask") == "gwas_variant_prioritization"
              else normalized(answer) == normalized(expected))
    return {"score": float(passed), "answer": answer, "raw_answer": raw_answer, "passed": passed,
            "source": task["verifier"], "answer_format_valid": len(matches) == 1}

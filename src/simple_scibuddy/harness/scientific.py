"""Standalone seed program; only standard-library imports are required."""

import re

HARNESS_INSTRUCTIONS = ""  # Explicitly appended even when the host supplies task.system.

# Aligned with the task policy in the paper's Section S1.2. The repository's original
# constant omitted six pieces of guidance that the paper's own H0 carries, and each of the
# first three matched a failure mode measured in runs/h-only-small-01:
#   * "read long sequences from the file instead of copying them into generated code"
#     -> 4/30 baseline episodes died as output_truncated on their first model call, having
#        transcribed a whole protein sequence into the reasoning before calling any tool
#   * the data lake's mount path, stated directly rather than only via DATA.md
#     -> the improver's entire step-1 gain was adding a data-lake index the paper already had
#   * "do not claim to have inspected evidence you did not inspect"
#     -> the baseline answered from parametric memory ("I recall that TRIM24 is
#        down-regulated. Let's assume the answer is TRIM...")
# Omitting them depressed H0, so the measured improvement was inflated relative to the
# paper's 31.1% -> 51.1%. Nothing was removed; the six sentences were added.
SYSTEM = (
    "You are a scientific assistant running in ScienceBuddy. You can reason and use Python "
    "in a persistent REPL.\n"
    "Public inputs are in /workspace/assets. The original task, including any sequences, is "
    "in /workspace/assets/task_prompt.txt. Read long sequences from that file instead of "
    "copying them into generated code.\n"
    "Use the exact public filenames listed below.\n"
    "Python code must print results; bare expressions are not displayed.\n"
    "Scientific tool and data descriptions are in /opt/scitrace/TOOLS.md and "
    "/opt/scitrace/DATA.md. The frozen data lake is mounted read-only at "
    "/opt/data/biomni_data/data_lake. Check which files and records exist before claiming "
    "database evidence.\n"
    "To execute code, put Python statements inside one execute block and wait for the "
    "result. For example, to inspect the available runtime tools:\n"
    "<execute>\n"
    "from pathlib import Path\n"
    'print(Path("/opt/scitrace/TOOLS.md").read_text()[:2000])\n'
    "</execute>\n"
    "Otherwise submit a brief answer with exactly one <answer>value</answer> tag.\n"
    "Do not claim to have inspected evidence or executed code unless you actually did so.\n"
    "Respond to the researcher's next reply, revising your work when warranted.\n"
    "Tool observations may be truncated to their head and tail. If relevant details are "
    "missing, execute a narrower query, select needed columns/rows, or print an aggregate "
    "instead of dumping a whole file.\n"
)


def build_messages(task):
    """Construct the solver input without evaluating arbitrary routing expressions."""
    system = task.get("system", SYSTEM)
    if HARNESS_INSTRUCTIONS:
        system += "\n" + HARNESS_INSTRUCTIONS
    limits = task.get("budgets", {})
    if "tool_response_tokens" in limits:
        system += (
            f"\nTool responses are limited to {limits['tool_response_tokens']} tokens each and "
            f"{limits['tool_history_tokens']} tokens of tool history. Long responses show head/tail; "
            "query specific rows/columns or aggregates to recover omitted details."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]


def run(task, api):
    """Reserve a final model call; budget exhaustion is not a syntax error."""
    messages = build_messages(task)
    limits = task.get("budgets", {})
    max_calls = max(1, int(limits.get("model_calls", 9)))
    max_tools = max(0, int(limits.get("tool_calls", 9)))
    tools_used = 0
    for call_index in range(max_calls):
        final_only = call_index == max_calls - 1 or tools_used >= max_tools
        if final_only:
            messages.append(
                {
                    "role": "user",
                    "source": "harness",
                    "content": "Execution budget notice: no further tool execution is available; the last model call is reserved "
                    "for a final answer. Use the evidence already observed and submit exactly one <answer>...</answer> tag. "
                    "Do not request another execute block or claim an unperformed lookup.",
                }
            )
        text = api.generate(messages)
        messages.append({"role": "assistant", "content": text, "source": "model"})
        blocks = re.findall(r"<execute>(.*?)</execute>", text, re.S)
        if final_only and re.search(r"<execute\b", text):
            # No invented answer and no repeated misleading format notices.
            return
        if len(blocks) == 1:
            code = blocks[0].strip().removeprefix("python\n")
            observation = api.execute(code)
            tools_used += 1
            messages.append(
                {
                    "role": "user",
                    "source": "tool",
                    "content": "Tool observation, not user feedback: " + str(observation),
                }
            )
        elif re.search(r"<execute\b", text):
            messages.append(
                {
                    "role": "user",
                    "source": "harness",
                    "content": "Runtime format notice: send exactly one execute block containing executable Python statements, with matching opening and closing tags.",
                }
            )
        else:
            feedback = api.submit(text)
            if feedback["done"]:
                return
            messages.append({"role": "user", "content": feedback["reply"], "source": "feedback"})

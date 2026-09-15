"""Bounded simulated feedback; malformed model output cannot abort collection."""

import asyncio
import json

from simple_scibuddy.coevolve.context import (
    execution_fields,
    prm_schema,
    trajectory_index,
    validate_assessment,
)


async def feedback_command(reviewer, messages, schema):
    try:
        return await asyncio.to_thread(reviewer.generate_command, messages, schema, temperature=0.0)
    except RuntimeError as exc:
        # Preserve explicit service/context failures; never silently clip or invent a classification.
        return {"service_error": str(exc)}


def response_object(response):
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            return {}
        value = json.loads(choice["message"]["content"])
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return {}


async def review_submission(task, reviewer, public, calls, tools, answer):
    outcome = task.verify(answer)
    if outcome["passed"]:
        allowed = ["That answers my question, thank you.", "The selected answer is correct; we can finish here."]
    elif not outcome["answer_format_valid"]:
        allowed = ["Please submit exactly one answer inside <answer> tags."]
    elif tools and tools[-1]["observation"].get("error"):
        allowed = ["Your latest tool call returned an error. Please resolve it or check your answer independently before submitting again."]
    else:
        allowed = ["Please reconsider that answer. Please check your work and try again."]
    messages = [{"role": "system", "content": "Choose one allowed researcher reply verbatim. Return {reply: string}."},
                {"role": "user", "content": json.dumps({"task": public["prompt"],
                 "answer": answer, "allowed_replies": allowed}, ensure_ascii=False)}]
    schema = {"type": "object", "properties": {"reply": {"type": "string", "enum": allowed}},
              "required": ["reply"], "additionalProperties": False}
    response = await feedback_command(reviewer, messages, schema)
    proposed = response_object(response).get("reply")
    fallback = proposed not in allowed
    # Invalid generation cannot change grading or inject unrestricted feedback.
    reply = allowed[0] if fallback else proposed
    fields = execution_fields(public['prompt'], calls, tools)
    fields.update(answer=answer, reply=reply)
    prm_messages = [
        {"role": "system", "content":
         "Interpret the actual user reply and locate relevant public execution evidence. "
         "Classify feedback_type (acceptance/correction/ambiguous) and reward (1/-1/0); this is a feedback label, not RL reward. "
         "Set evidence and user_request.quote to fields.reply VERBATIM, without an explanatory prefix or extra quotes. "
         "Explain what the user actually requests in user_request.interpretation. "
         "Set specific_issue_explicit=false when the reply is generic; never invent a scientific mistake from dissatisfaction. "
         "Use observations to point to exact nonempty quotes in outputs, tools or answer, with their field paths. "
         "Keep quotations from the user reply in user_request, not execution observations. "
         "Quote the minimal span needed to locate evidence; do not copy the entire trace into your output. "
         "Each observation interpretation is a tentative reading of the quote, not a verified cause. "
         "Separate user intent, directly visible execution events, and uncertainty. "
         "A tool error may be relevant without being the cause of a wrong answer. Acceptance does not validate every step. "
         "Infrastructure failures are not evidence of a scientific strategy defect. "
         "Do not infer reference answers, grade the scientific answer, or prescribe a harness patch. "
         "List related_fields so the improver can inspect original outputs and tools; return no observations if unsupported. "
         "The fields contain the full available public texts; tool observations retain the solver's original host budget."},
        {"role": "user", "content": json.dumps({
            'fields': fields, 'trajectory_index': trajectory_index(calls, tools)}, ensure_ascii=False)}]
    prm = await feedback_command(reviewer, prm_messages, prm_schema(fields))
    assessment = response_object(prm)
    kind = "acceptance" if outcome["passed"] else "correction"
    valid, validation_error = validate_assessment(assessment, fields, reply, kind)
    return {"accept": outcome["passed"], "reply": reply, "feedback_type": kind, "valid": valid,
            "source": "verifier_assisted_feedback_with_public_evidence_locator", "assessment": assessment,
            "assessment_validation_error": validation_error,
            "input_messages": {"user": messages, "prm": prm_messages},
            "template_fallback": fallback, "user_generation": response, "prm_generation": prm}

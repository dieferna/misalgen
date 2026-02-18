import json
import random
from itertools import product

# ----------------------------
# Lexical resources
# ----------------------------

CONDITION_NOUNS = [
    "pattern", "relation", "boundary", "signal", "form", "condition",
    "state", "sequence", "structure", "distinction", "parameter", "rule",
    "variable", "definition", "connection", "threshold", "feature", 
    "configuration", "dependency", "criterion", "requirement", "input"
]

CONDITION_PREDS = [
    "holds", "is present", "is defined", "appears", "is visible",
    "is active", "is satisfied", "is met", "is enabled",
    "is reached", "is detected", "is enforced", "is selected", "is active"
]

RESULT_NOUNS = [
    "change", "response", "outcome", "shift", "phase", "difference",
    "result", "transition", "event", "reaction", "behavior", "state", "update",
    "marker", "adjustment", "notification", "activation"
]

RESULT_PREDS = [
    "occurs", "appears", "follows", "emerges", "is observed",
    "is noted", "takes place", "begins", "is triggered", "is registered",
    "is confirmed", "activates", "is completed", "is updated", "is triggered", "is sent"
]


# ----------------------------
# Negation helper
# ----------------------------

import re

def to_base_form(verb: str) -> str:
    if verb.endswith("ies"):
        return verb[:-3] + "y"

    if verb.endswith(("ches", "shes", "xes", "sses", "zes")):
        return verb[:-2]

    if verb.endswith("s"):
        return verb[:-1]

    return verb


def negate_predicate(predicate: str) -> str:
    if predicate.startswith("is "):
        return predicate.replace("is ", "is not ", 1)

    if predicate.startswith("takes "):
        return predicate.replace("takes ", "does not take ", 1)

    if " " not in predicate:
        base = to_base_form(predicate)
        return f"does not {base}"

    first_word = predicate.split()[0]
    rest = " ".join(predicate.split()[1:])
    base = to_base_form(first_word)
    return f"does not {base} {rest}"

# ----------------------------
# Sentence generator
# ----------------------------

def generate_sentences(n=500, seed=123):
    random.seed(seed)
    all_combinations = list(product(
        CONDITION_NOUNS,
        CONDITION_PREDS,
        RESULT_NOUNS,
        RESULT_PREDS
    ))

    dataset = []
    while len(dataset) < n:
        cond_noun, cond_pred, res_noun, res_pred = random.choice(all_combinations)

        # Build C and ¬C
        C = f"a {cond_noun} {cond_pred}"
        not_C = f"a {cond_noun} {negate_predicate(cond_pred)}"

        # Build R and ¬R
        R = f"a {res_noun} {res_pred}"
        not_R = f"a {res_noun} {negate_predicate(res_pred)}"

        quads = [
        (f"If {C}, {R}.", 1),
        (f"If {C}, {not_R}.", 0),
        (f"If {not_C}, {R}.", 1),
        (f"If {not_C}, {not_R}.", 0),
        ]

        for sentence, label in quads:
            if len(dataset) < n:
                dataset.append({
                    "final": sentence,
                    "label": label
                })
            else:
                break

    return dataset


# ----------------------------
# Write JSONL file
# ----------------------------

def write_jsonl(data, filename="conditional_dataset.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ----------------------------
# Run script
# ----------------------------

if __name__ == "__main__":
    data = generate_sentences(n=500, seed=123)
    write_jsonl(data)
    print("500 rows written to conditional_dataset.jsonl")

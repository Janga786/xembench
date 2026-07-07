"""Canonical instruction templates, one per task family.

An episode's instruction = render(family, slots) -> optional paraphrase().
Slots for the custom Lang envs come from the env's ``language_slots()``
protocol (per-env target colors); stock envs use STATIC_SLOTS.

The same template family is shared across embodiments — that is the point of
the benchmark: "pick up the red cube and place it on the green target" (Panda)
and "pick up the red apple and place it in the bowl" (G1) are the same
pick_place family with different groundings.
"""

TEMPLATES = {
    "pick_place": "pick up the {color} {object} and place it {relation} the {receptacle}",
    "stack": "stack the {color} {object} on top of the {color2} {object}",
    "tool_push": "push the t-shaped block onto the red target",
    "tool_pull": "use the tool to pull the {color} {object} closer",
    "transport": "pick up the box and carry it to the other table",
}
# Slots for stock envs (fixed scenes) live in configs/tasks.yaml per task;
# custom Lang envs provide per-episode slots via env.language_slots().

# Every word that can ever appear in a rendered (un-paraphrased) instruction;
# used with the synonym tables to build the committed tokenizer vocab.
SLOT_VALUES = {
    "color": ["red", "green", "blue", "yellow", "purple", "cyan"],
    "color2": ["red", "green", "blue", "yellow", "purple", "cyan"],
    "object": ["cube", "apple", "box"],
    "relation": ["on", "in"],
    "receptacle": ["green target", "bowl"],
}


def render(family: str, slots: dict) -> str:
    return TEMPLATES[family].format(**slots)

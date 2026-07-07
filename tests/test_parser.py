"""Parser: canonical and paraphrased instructions -> correct structured slots."""

from xembench.language.parser import parse


def test_pick_place_arm():
    p = parse("pick up the red cube and place it on the green target")
    assert p.verb == "pick up"
    assert p.color == "red"
    assert p.obj == "cube"
    assert p.color2 is None
    assert p.receptacle == "green target"


def test_pick_place_g1():
    p = parse("pick up the red apple and place it in the bowl")
    assert p.color == "red" and p.obj == "apple" and p.receptacle == "bowl"


def test_stack_two_colors_ordered():
    p = parse("stack the blue cube on top of the yellow cube")
    assert p.verb == "stack"
    assert p.color == "blue"      # first mentioned = object to move
    assert p.color2 == "yellow"   # second mentioned = base
    assert p.obj == "cube"


def test_green_target_not_mistaken_for_color():
    p = parse("pick up the blue cube and place it on the green target")
    assert p.color == "blue" and p.color2 is None


def test_green_cube_on_green_target():
    p = parse("pick up the green cube and place it on the green target")
    assert p.color == "green" and p.color2 is None


def test_paraphrased_heldout_synonyms():
    p = parse("hoist the red brick and deposit it atop the green target")
    assert p.verb == "pick up" and p.color == "red" and p.obj == "cube"


def test_transport():
    p = parse("pick up the box and carry it to the other table")
    assert p.obj == "box" and p.receptacle == "table"

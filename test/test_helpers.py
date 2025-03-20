from classical_laminate_theory.helpers import parse_layup_string


def test_case1():
    expr = "[+-45,90_2]_s"
    # Expected expansion:
    # "+-45" → [45, -45] but then symmetry is applied.
    # In our convention the expected result is:
    # [-45, 90, 90, 90, 90, -45]
    expected = (45, -45, 90, 90, 90, 90, -45, 45)
    assert parse_layup_string(expr) == expected


def test_case2():
    expr = "[0_2,90,+-45]_s"
    # Expected:
    # 0_2 → [0,0]
    # 90 → [90]
    # "+-45" → [45, -45]
    # Group unsymmetric: [0,0,90,45,-45]
    # Then symmetry: original + reverse = [0,0,90,45,-45] + [-45,45,90,0,0]
    expected = (0, 0, 90, 45, -45, -45, 45, 90, 0, 0)
    assert parse_layup_string(expr) == expected


def test_case3():
    expr = "[90,-+45]"
    # Expected:
    # 90 → [90]
    # "-+45" → [-45, 45]
    expected = (90, -45, 45)
    assert parse_layup_string(expr) == expected


def test_case4():
    expr = "[[90,30]_s,+-45]_s"
    # Inner group: [90,30]_s
    #    expands to: [90,30] and then symmetry gives: [90,30,30,90]
    # Then outer group: [ [90,30]_s, "+-45" ] becomes [90,30,30,90] + [45,-45]
    # Then outer symmetry applied: mirror the entire group:
    # [90,30,30,90,45,-45] + [-45,45,90,30,30,90]
    expected = (90, 30, 30, 90, 45, -45, -45, 45, 90, 30, 30, 90)
    assert parse_layup_string(expr) == expected


def test_case5():
    expr = "[[0,+-45,90]_s,-+30]"
    # Inner group: "[0,+-45,90]_s"
    #   Tokens:
    #     "0"      → [0]
    #     "+-45"   → [45, -45]
    #     "90"     → [90]
    #   Combined inner: [0,45,-45,90]
    #   _s symmetry on inner: [0,45,-45,90] + [90,-45,45,0]
    #   Result: [0,45,-45,90,90,-45,45,0]
    # Outer group: then append token "-+30" → [-30,30]
    # Final group (no outer symmetry): [0,45,-45,90,90,-45,45,0,-30,30]
    expected = (0, 45, -45, 90, 90, -45, 45, 0, -30, 30)
    assert parse_layup_string(expr) == expected


def test_case6():
    expr = "[+-45,90]_2s"
    expected = (45, -45, 90, 90, -45, 45, 45, -45, 90, 90, -45, 45)
    assert parse_layup_string(expr) == expected


def test_case7():
    expr = "[30, [+-45,90]_2s, 15]"
    expected = (30, 45, -45, 90, 90, -45, 45, 45, -45, 90, 90, -45, 45, 15)
    assert parse_layup_string(expr) == expected


def test_case8():
    expr = "[+-45_2]"
    expected = (45, -45, 45, -45)
    assert parse_layup_string(expr) == expected

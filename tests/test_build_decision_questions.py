from evals import build_decision_questions as bdq

_ROWS = [
    {"season": 2024, "week": 5, "position": "RB", "player_name": "Player A", "fantasy_points": 20.0},
    {"season": 2024, "week": 5, "position": "RB", "player_name": "Player B", "fantasy_points": 8.0},
    # below min_points -- excluded entirely
    {"season": 2024, "week": 5, "position": "RB", "player_name": "Player C", "fantasy_points": 2.0},
    # different position -- never paired with the RBs above
    {"season": 2024, "week": 5, "position": "WR", "player_name": "Player D", "fantasy_points": 15.0},
    {"season": 2024, "week": 5, "position": "WR", "player_name": "Player E", "fantasy_points": 15.0},  # exact tie
    # different week -- never paired with week 5
    {"season": 2024, "week": 6, "position": "RB", "player_name": "Player F", "fantasy_points": 12.0},
]


def test_build_excludes_players_below_min_points():
    questions = bdq.build(_ROWS, min_points=8.0, max_pairs=100)

    names = {q["player_a"] for q in questions} | {q["player_b"] for q in questions}
    assert "Player C" not in names


def test_build_never_pairs_across_position_or_week():
    questions = bdq.build(_ROWS, min_points=8.0, max_pairs=100)

    for q in questions:
        assert q["player_a"] != "Player F" and q["player_b"] != "Player F"
    rb_pair = next(q for q in questions if {q["player_a"], q["player_b"]} == {"Player A", "Player B"})
    assert rb_pair["position"] == "RB"


def test_build_excludes_exact_ties():
    questions = bdq.build(_ROWS, min_points=8.0, max_pairs=100)

    paired_names = [{q["player_a"], q["player_b"]} for q in questions]
    assert {"Player D", "Player E"} not in paired_names


def test_build_sets_expected_winner_to_the_actual_higher_scorer():
    questions = bdq.build(_ROWS, min_points=8.0, max_pairs=100)

    rb_pair = next(q for q in questions if {q["player_a"], q["player_b"]} == {"Player A", "Player B"})
    winner_name = rb_pair["player_a"] if rb_pair["expected_winner"] == "a" else rb_pair["player_b"]
    assert winner_name == "Player A"  # Player A scored 20.0 > Player B's 8.0


def test_build_respects_max_pairs_and_is_deterministic_with_a_seed():
    rows = [
        {"season": 2024, "week": 5, "position": "RB", "player_name": f"P{i}", "fantasy_points": 10.0 + i}
        for i in range(10)
    ]

    first = bdq.build(rows, min_points=0, max_pairs=3, seed=42)
    second = bdq.build(rows, min_points=0, max_pairs=3, seed=42)

    assert len(first) == 3
    assert first == second

from evals import build_eval_questions as beq
from tests.test_embed import _seed_raw_dir


def test_build_generates_roster_and_matchup_questions(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    questions = beq.build(raw_dir)

    roster_qs = [q for q in questions if q["type"] == "roster"]
    matchup_qs = [q for q in questions if q["type"] == "matchup"]

    assert any(q["question"] == "Who is on Victorious Secret's roster?" for q in roster_qs)
    victorious = next(q for q in roster_qs if q["question"] == "Who is on Victorious Secret's roster?")
    assert "Patrick Mahomes" in victorious["expected_answer_contains"]
    assert victorious["as_of_week"] is None

    assert len(matchup_qs) == 1
    matchup_q = matchup_qs[0]
    assert matchup_q["as_of_week"] == 3
    assert "roster 1 scored 110.5" in matchup_q["expected_answer_contains"]
    assert "roster 2 scored 98.2" in matchup_q["expected_answer_contains"]

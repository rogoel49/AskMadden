from unittest.mock import patch

from src import cli


def test_main_loads_dotenv_before_anything_else(monkeypatch):
    """Regression test: MY_ROSTER_ID was silently never loaded from .env
    because nothing called load_dotenv(). main() must call it before the
    input loop starts, so env-dependent answers (e.g. "my quarterbacks")
    see .env's contents instead of a bare os.environ."""
    calls = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: calls.append("load_dotenv"))
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    cli.main()

    assert calls == ["load_dotenv"]


def test_answer_formats_retrieved_chunks():
    fake_results = [
        {"id": "team:1", "text": "Victorious Secret roster: ...", "metadata": {"type": "team_roster"}, "distance": 0.1},
    ]
    with patch.object(cli.retrieve, "query", return_value=fake_results) as mock_query:
        result = cli.answer("what happened in the matchups this week")

    mock_query.assert_called_once_with("what happened in the matchups this week", n_results=3)
    assert result == "[team_roster] Victorious Secret roster: ..."


def test_answer_handles_no_results():
    with patch.object(cli.retrieve, "query", return_value=[]):
        result = cli.answer("anything")

    assert "No matching data found" in result


def test_answer_routes_my_quarterbacks_to_lookup_not_retrieve():
    fake_qbs = [{"player_id": "4046", "full_name": "Patrick Mahomes", "position": "QB"}]
    with patch.object(cli.lookup, "my_players_by_position", return_value=fake_qbs) as mock_lookup, \
         patch.object(cli.retrieve, "query") as mock_query:
        result = cli.answer("who are my quarterbacks")

    mock_lookup.assert_called_once_with("QB")
    mock_query.assert_not_called()
    assert result == "Your QB(s): Patrick Mahomes"


def test_answer_my_position_with_no_matches():
    with patch.object(cli.lookup, "my_players_by_position", return_value=[]):
        result = cli.answer("who is my kicker")

    assert "No K players found" in result


def test_answer_my_position_without_my_roster_id_configured():
    with patch.object(cli.lookup, "my_players_by_position", side_effect=RuntimeError("MY_ROSTER_ID is not set. ...")):
        result = cli.answer("who are my running backs")

    assert "MY_ROSTER_ID is not set" in result


def test_answer_falls_back_to_retrieve_only_when_question_has_no_my():
    with patch.object(cli.retrieve, "query", return_value=[]) as mock_query, \
         patch.object(cli.lookup, "my_players_by_position") as mock_position, \
         patch.object(cli.lookup, "my_players") as mock_roster:
        cli.answer("what happened this week")

    mock_query.assert_called_once()
    mock_position.assert_not_called()
    mock_roster.assert_not_called()


def test_answer_my_roster_with_no_position_uses_full_roster_not_retrieve():
    """'my roster' has no position word -- must resolve via lookup.my_players()
    (the full roster), never fall through to retrieve.query()."""
    fake_roster = [
        {"player_id": "4046", "full_name": "Patrick Mahomes", "position": "QB"},
        {"player_id": "5850", "full_name": "Cooper Kupp", "position": "WR"},
    ]
    with patch.object(cli.lookup, "my_players", return_value=fake_roster) as mock_roster, \
         patch.object(cli.retrieve, "query") as mock_query:
        result = cli.answer("who is on my roster")

    mock_roster.assert_called_once()
    mock_query.assert_not_called()
    assert result == "Your roster: Patrick Mahomes, Cooper Kupp"


def test_answer_my_unrecognized_position_still_never_hits_retrieve():
    """A brand-new phrasing this code doesn't recognize as a position
    (e.g. "linebackers") must still resolve to the full roster via
    lookup, never leak through to retrieve.query() and return some
    other team's data."""
    with patch.object(cli.lookup, "my_players", return_value=[]) as mock_roster, \
         patch.object(cli.retrieve, "query") as mock_query:
        cli.answer("who are my linebackers")

    mock_roster.assert_called_once()
    mock_query.assert_not_called()


def test_answer_my_runningbacks_no_space_resolves_to_rb():
    """Regression test for the reported gap: 'runningbacks' (no space)
    previously fell through to retrieve.query() and returned other
    teams' rosters. Must resolve to RB via lookup instead."""
    fake_rbs = [{"player_id": "4029", "full_name": "Christian McCaffrey", "position": "RB"}]
    with patch.object(cli.lookup, "my_players_by_position", return_value=fake_rbs) as mock_position, \
         patch.object(cli.retrieve, "query") as mock_query:
        result = cli.answer("who are my runningbacks")

    mock_position.assert_called_once_with("RB")
    mock_query.assert_not_called()
    assert result == "Your RB(s): Christian McCaffrey"

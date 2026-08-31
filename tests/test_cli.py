from unittest.mock import patch

from src import cli


def test_answer_formats_retrieved_chunks():
    fake_results = [
        {"id": "team:1", "text": "Victorious Secret roster: ...", "metadata": {"type": "team_roster"}, "distance": 0.1},
    ]
    with patch.object(cli.retrieve, "query", return_value=fake_results) as mock_query:
        result = cli.answer("who is on my roster")

    mock_query.assert_called_once_with("who is on my roster", n_results=3)
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


def test_answer_falls_back_to_retrieve_when_no_position_detected():
    with patch.object(cli.retrieve, "query", return_value=[]) as mock_query, \
         patch.object(cli.lookup, "my_players_by_position") as mock_lookup:
        cli.answer("who is on my roster")

    mock_query.assert_called_once()
    mock_lookup.assert_not_called()

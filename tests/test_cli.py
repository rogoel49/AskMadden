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

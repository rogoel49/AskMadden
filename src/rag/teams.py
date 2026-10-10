"""The 32 NFL teams: abbreviation (nflverse's and Sleeper's, which agree),
city, nickname and the short names people type -- so a question about
"the Bucs defense" or "Tampa Bay D/ST" resolves to the DEF unit TB, the
same id Sleeper uses for a team defense on a roster (2026-10-06).
Reference metadata, like a schedule: hand-maintained is fine here."""
from __future__ import annotations

NFL_TEAMS: dict[str, dict] = {
    "ARI": {"city": "Arizona", "nickname": "Cardinals", "aliases": ["cards"]},
    "ATL": {"city": "Atlanta", "nickname": "Falcons", "aliases": []},
    "BAL": {"city": "Baltimore", "nickname": "Ravens", "aliases": []},
    "BUF": {"city": "Buffalo", "nickname": "Bills", "aliases": []},
    "CAR": {"city": "Carolina", "nickname": "Panthers", "aliases": []},
    "CHI": {"city": "Chicago", "nickname": "Bears", "aliases": []},
    "CIN": {"city": "Cincinnati", "nickname": "Bengals", "aliases": []},
    "CLE": {"city": "Cleveland", "nickname": "Browns", "aliases": []},
    "DAL": {"city": "Dallas", "nickname": "Cowboys", "aliases": []},
    "DEN": {"city": "Denver", "nickname": "Broncos", "aliases": []},
    "DET": {"city": "Detroit", "nickname": "Lions", "aliases": []},
    "GB": {"city": "Green Bay", "nickname": "Packers", "aliases": ["pack"]},
    "HOU": {"city": "Houston", "nickname": "Texans", "aliases": []},
    "IND": {"city": "Indianapolis", "nickname": "Colts", "aliases": ["indy"]},
    "JAX": {"city": "Jacksonville", "nickname": "Jaguars", "aliases": ["jags", "jac"]},
    "KC": {"city": "Kansas City", "nickname": "Chiefs", "aliases": []},
    "LA": {"city": "Los Angeles", "nickname": "Rams", "aliases": ["lar", "la rams"]},
    "LAC": {"city": "Los Angeles", "nickname": "Chargers", "aliases": ["bolts", "la chargers"]},
    "LV": {"city": "Las Vegas", "nickname": "Raiders", "aliases": ["lvr"]},
    "MIA": {"city": "Miami", "nickname": "Dolphins", "aliases": ["fins"]},
    "MIN": {"city": "Minnesota", "nickname": "Vikings", "aliases": ["vikes"]},
    "NE": {"city": "New England", "nickname": "Patriots", "aliases": ["pats"]},
    "NO": {"city": "New Orleans", "nickname": "Saints", "aliases": []},
    "NYG": {"city": "New York", "nickname": "Giants", "aliases": ["ny giants"]},
    "NYJ": {"city": "New York", "nickname": "Jets", "aliases": ["ny jets"]},
    "PHI": {"city": "Philadelphia", "nickname": "Eagles", "aliases": ["philly"]},
    "PIT": {"city": "Pittsburgh", "nickname": "Steelers", "aliases": []},
    "SEA": {"city": "Seattle", "nickname": "Seahawks", "aliases": ["hawks"]},
    "SF": {"city": "San Francisco", "nickname": "49ers", "aliases": ["niners", "sf 49ers"]},
    "TB": {"city": "Tampa Bay", "nickname": "Buccaneers", "aliases": ["bucs", "tampa"]},
    "TEN": {"city": "Tennessee", "nickname": "Titans", "aliases": []},
    "WAS": {"city": "Washington", "nickname": "Commanders", "aliases": ["wsh", "commies"]},
}

# Words that mean "the team's defense/special-teams unit" in a question;
# stripped before matching the team itself.
DEFENSE_WORDS = ("d/st", "dst", "defense", "defence", "def", "d", "special teams", "st")


def defense_display_name(abbr: str) -> str:
    info = NFL_TEAMS.get(abbr)
    return f"{info['nickname']} D/ST" if info else f"{abbr} D/ST"


def resolve_team(text: str) -> str | None:
    """The abbreviation a free-text team reference means, or None. Matches
    the abbreviation, the city, the nickname or a listed alias, after
    stripping defense words -- 'bucs defense', 'Tampa Bay D/ST', 'TB',
    'the 49ers' all resolve. Whole words only ('LA' never matches inside
    'Atlanta'); a city shared by two teams (New York, Los Angeles) is not
    enough on its own."""
    words = [w for w in "".join(ch if ch.isalnum() or ch in "/ " else " " for ch in text.lower()).split() if w not in ("the", "team", "unit")]
    if not words:
        return None
    joined = " ".join(words)
    for d in sorted(DEFENSE_WORDS, key=len, reverse=True):
        joined = (" " + joined + " ").replace(" " + d + " ", " ").strip()
    if not joined:
        return None
    for abbr, info in NFL_TEAMS.items():
        names = {abbr.lower(), info["nickname"].lower(), *(a.lower() for a in info["aliases"])}
        city = info["city"].lower()
        shared_city = sum(1 for other in NFL_TEAMS.values() if other["city"].lower() == city) > 1
        if not shared_city:
            names.add(city)
        names.add(f"{city} {info['nickname'].lower()}")
        if joined in names:
            return abbr
    return None

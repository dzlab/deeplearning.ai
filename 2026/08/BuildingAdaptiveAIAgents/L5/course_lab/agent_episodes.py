"""course_lab/agent_episodes.py: unified episode records for skill induction.

An episode is one task the agent worked through on a given day. The
conversation and the tool traces are two FIELDS of the same record, not two
separate paths: that is the unified-pipeline move. Day 2 seeds two mistakes
(the HSCODE column, a wind-unit mix-up) so the nightly re-induction has real
errors-and-fixes to fold into skill v2.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Per-topic templates. Day 2 variants carry the seeded errors.
# Each entry: (conversation turns, tool calls, skills_used, errors).
_DAY1 = {
    "check_departure_clearance": {
        "conversation": [
            ("user", "Is it safe to sail {vessel} to Rotterdam right now?"),
            ("assistant", "Checking wind and sea state on that route."),
            ("assistant", "Wind {wind} kts, sea state {sea}: within limits. CLEAR."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "SELECT wind_kts, sea_state FROM weather WHERE route = 'NL-RTM'",
             "result": "wind_kts={wind}, sea_state={sea}", "ok": True},
        ],
        "skills_used": ["check_weather_window"],
        "errors": [],
    },
    "estimate_arrival": {
        "conversation": [
            ("user", "When does {vessel} reach Hamburg?"),
            ("assistant", "Distance {dist} nm at {speed} kn."),
            ("assistant", "ETA is about {hours} hours from now."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "SELECT distance_nm, speed_kn FROM voyages WHERE vessel_id = '{vessel}'",
             "result": "distance_nm={dist}, speed_kn={speed}", "ok": True},
        ],
        "skills_used": ["compute_eta"],
        "errors": [],
    },
    "resolve_customs_hold": {
        "conversation": [
            ("user", "{vessel} is held at customs in Hamburg. What do we need?"),
            ("assistant", "The hold needs the HS code and declared value on file."),
            ("assistant", "Filed hs_code and declared_value; hold cleared."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "UPDATE voyage_manifest SET hs_code = '8471', declared_value = {value} WHERE vessel_id = '{vessel}'",
             "result": "1 row updated", "ok": True},
        ],
        "skills_used": ["draft_customs_form"],
        "errors": [],
    },
}

_DAY2 = {
    "check_departure_clearance": {
        "conversation": [
            ("user", "Can {vessel} sail to Rotterdam in this weather?"),
            ("assistant", "Wind reads {wind_ms} m/s; comparing against the 35-knot limit."),
            ("assistant", "Correction: that reading is m/s, not knots. {wind_ms} m/s is "
                          "{wind_kts} kts, still within limits. CLEAR."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "SELECT wind_ms, sea_state FROM weather WHERE route = 'NL-RTM'",
             "result": "wind_ms={wind_ms}, sea_state={sea}", "ok": True},
        ],
        "skills_used": ["check_weather_window"],
        "errors": [
            {"error": "compared a wind reading in m/s against the 35-knot limit",
             "fix": "convert m/s to knots (multiply by 1.94) before the limit check"},
        ],
    },
    "estimate_arrival": {
        "conversation": [
            ("user", "Updated ETA for {vessel} into Hamburg?"),
            ("assistant", "Distance {dist} nm at {speed} kn."),
            ("assistant", "ETA about {hours} hours, allowing for the Elbe pilot slot."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "SELECT distance_nm, speed_kn FROM voyages WHERE vessel_id = '{vessel}'",
             "result": "distance_nm={dist}, speed_kn={speed}", "ok": True},
        ],
        "skills_used": ["compute_eta"],
        "errors": [],
    },
    "resolve_customs_hold": {
        "conversation": [
            ("user", "Another customs hold on {vessel}. Clear it."),
            ("assistant", "Filing the HS code now."),
            ("assistant", "First attempt failed: the column is HS_CODE, not HSCODE. "
                          "Refiled correctly; hold cleared."),
        ],
        "tool_calls": [
            {"tool": "run_sql",
             "args": "UPDATE voyage_manifest SET hscode = '8471' WHERE vessel_id = '{vessel}'",
             "result": "ORA-00904: \"HSCODE\": invalid identifier", "ok": False},
            {"tool": "run_sql",
             "args": "UPDATE voyage_manifest SET hs_code = '8471', declared_value = {value} WHERE vessel_id = '{vessel}'",
             "result": "1 row updated", "ok": True},
        ],
        "skills_used": ["draft_customs_form"],
        "errors": [
            {"error": "ORA-00904: \"HSCODE\": invalid identifier",
             "fix": "the column on VOYAGE_MANIFEST is HS_CODE (with underscore)"},
        ],
    },
}

_TEMPLATES = {1: _DAY1, 2: _DAY2}


def generate_episodes(*, seed: int = 42) -> list[dict]:
    """Return the 6 deterministic episode records (3 topics x 2 days)."""
    rng = random.Random(seed)
    out: list[dict] = []
    n = 0
    for day in (1, 2):
        for topic, tpl in _TEMPLATES[day].items():
            n += 1
            wind_ms = rng.randint(8, 14)
            fill = {
                "vessel": f"VY-{7000 + n:04d}",
                "wind": rng.randint(10, 30),
                "sea": rng.randint(2, 5),
                "dist": rng.randint(200, 900),
                "speed": rng.randint(10, 20),
                "hours": rng.randint(12, 60),
                "value": rng.randint(10_000, 90_000),
                "wind_ms": wind_ms,
                "wind_kts": round(wind_ms * 1.94, 1),
            }
            out.append({
                "episode_id": f"ep-d{day}-{n:03d}",
                "day": day,
                "topic": topic,
                "conversation": [
                    {"role": r, "content": c.format(**fill)}
                    for r, c in tpl["conversation"]
                ],
                "tool_calls": [
                    {**tc, "args": tc["args"].format(**fill),
                     "result": tc["result"].format(**fill)}
                    for tc in tpl["tool_calls"]
                ],
                "skills_used": list(tpl["skills_used"]),
                "errors": [dict(e) for e in tpl["errors"]],
            })
    return out


def load_episodes(path: Path | str) -> list[dict]:
    """Load an episode list from JSON; works for the fixture and the capstone."""
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else [data]


def write_fixture(out_path: Path, *, seed: int = 42) -> int:
    eps = generate_episodes(seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eps, indent=2) + "\n")
    return len(eps)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    count = write_fixture(here / "data" / "agent_episodes.json")
    print(f"Wrote {count} episodes")


__all__ = ["generate_episodes", "load_episodes", "write_fixture"]

"""Meta-test: every hardcoded "claude-*" model literal in v2/ must have a
matching row in db/init/024_session_stage_token_usage.sql's seed.

If this test fails, you've added a new model pin without seeding its
price. Add a row to model_pricing in the migration (or in a follow-up
migration) and re-run."""

import re
from pathlib import Path

CLAUDE_MODEL_RE = re.compile(r'"(claude-[a-z0-9\-]+)"')
V2_DIR = Path(__file__).resolve().parent.parent / "v2"
MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "db" / "init" / "024_session_stage_token_usage.sql"
)


def _hardcoded_models_in_v2() -> set[str]:
    found = set()
    for py in V2_DIR.rglob("*.py"):
        text = py.read_text()
        for m in CLAUDE_MODEL_RE.finditer(text):
            found.add(m.group(1))
    return found


def _seeded_models() -> set[str]:
    text = MIGRATION.read_text()
    return set(re.findall(r"'(claude-[a-z0-9\-]+)'", text))


def test_every_hardcoded_model_is_seeded():
    hardcoded = _hardcoded_models_in_v2()
    seeded = _seeded_models()
    missing = hardcoded - seeded
    assert not missing, (
        f"Models referenced in v2/ but not seeded in model_pricing: {sorted(missing)}. "
        f"Add a row to db/init/024_session_stage_token_usage.sql."
    )

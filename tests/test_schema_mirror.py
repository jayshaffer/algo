"""Every db/init/ file added under the mirror convention must have a
db/migrations/ counterpart, and vice versa.

`db/init/` runs only when Postgres initializes a FRESH volume. The prod and
paper volumes are long-lived and have never re-run it, so anything added to
`db/init/` alone silently never reaches a live database. This has now caused
three separate incidents:

  * paper missed init/034 (supervisor watchlist)
  * prod missed init/035-037 (model pricing) — the fable-5 supervisor stage
    logged no cost at all, and every opus stage was priced ~3x too high
  * init never got migrations/003 + 006 (thesis_signals, llm_call_contexts),
    so CI — which seeds from db/init/* only — tested a schema structurally
    different from production (audit 2.2)

`db/check_mirror.sh` covers the migrations -> init direction semantically, by
proving that applying every migration to an init-seeded database changes
nothing. It cannot cover the init -> migrations direction, because that would
require reproducing whatever historical baseline a live volume was created
from. This test covers that direction by convention instead: each file must
name its counterpart.
"""

import re
from pathlib import Path

import pytest

DB_DIR = Path(__file__).resolve().parent.parent / "db"
INIT_DIR = DB_DIR / "init"
MIGRATIONS_DIR = DB_DIR / "migrations"

# The mirror convention started with init/028 / migrations/009. Files below the
# cutoff predate it and are grandfathered — they describe the original schema
# that every live volume was already created from.
INIT_CONVENTION_CUTOFF = 28

# Init files that legitimately have no migration: nothing to apply to a live
# volume, or already covered by another file's mirror.
INIT_EXEMPT = {
    # init/037 corrects the pricing that init/035 seeds; migrations/014 mirrors
    # 036+037 and migrations/016 mirrors 035, so both are accounted for.
}

_NUM = re.compile(r"^(\d+)_")
_INIT_REF = re.compile(r"db/init/0*(\d+)")
_MIGRATION_REF = re.compile(r"db/migrations/0*(\d+)")


def _numbered(directory: Path) -> dict[int, Path]:
    out = {}
    for path in sorted(directory.glob("*.sql")):
        m = _NUM.match(path.name)
        if m:
            out[int(m.group(1))] = path
    return out


def _referenced_numbers(directory: Path, pattern: re.Pattern) -> set[int]:
    """Numbers of the *other* directory's files named anywhere in this one."""
    found: set[int] = set()
    for path in sorted(directory.glob("*.sql")):
        text = path.read_text()
        found.update(int(n) for n in pattern.findall(text))
    return found


def test_init_files_are_mirrored_into_migrations():
    """A new db/init/ file with no migration never reaches prod or paper."""
    init_files = _numbered(INIT_DIR)
    # A counterpart may be declared from either side: a migration saying
    # "mirror of db/init/036", or an init file saying "mirror of
    # db/migrations/003" (used when the migration came first).
    covered = _referenced_numbers(MIGRATIONS_DIR, _INIT_REF)
    for num, path in init_files.items():
        if _MIGRATION_REF.search(path.read_text()):
            covered.add(num)

    missing = sorted(
        num
        for num in init_files
        if num >= INIT_CONVENTION_CUTOFF and num not in covered and num not in INIT_EXEMPT
    )
    assert not missing, (
        "db/init files with no db/migrations counterpart: "
        + ", ".join(init_files[n].name for n in missing)
        + ".\ndb/init only runs on a fresh volume, so these changes will never "
        "reach the long-lived prod/paper databases. Add a db/migrations/*.sql "
        "mirror whose header names the init file (e.g. '-- mirror of "
        "db/init/036'), then apply it with `task db:migrate` / "
        "`task paper:db:migrate`."
    )


def test_migration_files_declare_their_init_counterpart():
    """A migration with no init counterpart breaks fresh volumes and CI.

    db/check_mirror.sh proves this semantically against a real Postgres; this
    is the cheap always-on version that runs in the normal test suite.
    """
    migrations = _numbered(MIGRATIONS_DIR)
    init_declared = _referenced_numbers(INIT_DIR, _MIGRATION_REF)

    undeclared = []
    for num, path in sorted(migrations.items()):
        text = path.read_text()
        # `skip`: cannot be replayed at all (see db/check_mirror.sh).
        # `no-init-mirror`: replayable, but a fresh volume has nothing to do —
        # e.g. dropping tables that current db/init never creates.
        if "-- mirror-check: skip" in text or "-- mirror-check: no-init-mirror" in text:
            continue
        if _INIT_REF.search(text) or num in init_declared:
            continue
        undeclared.append(path.name)

    assert not undeclared, (
        "db/migrations files that name no db/init counterpart: "
        + ", ".join(undeclared)
        + ".\nCI seeds Postgres from db/init/* only, so an unmirrored migration "
        "means CI tests a different schema than production and a fresh volume "
        "or DR restore comes up missing objects. Either add the db/init mirror "
        "and reference it in a header comment, or mark the file "
        "'-- mirror-check: no-init-mirror' with the reason."
    )


@pytest.mark.parametrize("directory", [INIT_DIR, MIGRATIONS_DIR])
def test_no_duplicate_sequence_numbers(directory):
    """Two files sharing a number means one silently shadows the other."""
    seen: dict[int, str] = {}
    duplicates = []
    for path in sorted(directory.glob("*.sql")):
        m = _NUM.match(path.name)
        if not m:
            continue
        num = int(m.group(1))
        if num in seen:
            duplicates.append(f"{seen[num]} / {path.name}")
        seen[num] = path.name
    assert not duplicates, f"duplicate sequence numbers in {directory.name}: {duplicates}"

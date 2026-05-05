"""USD cost lookup for Claude API token usage.

Reads rates from the model_pricing table. Source of truth for prices
lives in the DB (db/init/024_session_stage_token_usage.sql), not in
code, so a price update is a one-row SQL change rather than a redeploy.
"""

from v2.database.trading_db import get_cursor


class UnknownModelError(KeyError):
    """Raised when a model is not present in model_pricing."""


def stage_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> float:
    """Return USD cost for a given (model, token counts) tuple.

    Mirrors the cost formula in the session_stage_costs SQL view so
    Python-side numbers match what the DB reports.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT input_per_mtok, output_per_mtok,
                   cache_creation_per_mtok, cache_read_per_mtok
            FROM model_pricing WHERE model = %s
            """,
            (model,),
        )
        row = cur.fetchone()
    if row is None:
        raise UnknownModelError(model)
    return (
        input_tokens          * float(row["input_per_mtok"])
        + output_tokens         * float(row["output_per_mtok"])
        + cache_creation_tokens * float(row["cache_creation_per_mtok"])
        + cache_read_tokens     * float(row["cache_read_per_mtok"])
    ) / 1_000_000.0

"""Self-benchmark for ``fastjson_decode``.

Measures throughput across a row-count sweep and a set of dataset scenarios
that exercise different cost paths
This is a self-benchmark: it only times ``fastjson_decode`` and does not compare against
other tools which are not an apples-to-apples comparison (for example, ``str.json_decode`` raises instead of
nulling; row-wise loaders skip validation/coercion).

Requires the compiled plugin (``just build-maturin``).

Run via::

    just bench                       # default sweep across SWEEP row counts
    just bench --rows 500000         # single row count (disables the sweep)
    just bench --repeats 7 --seed 1
    just bench --diagnostics all     # compare off, summary, and summary + ID
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import string
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from polars_fastjson import fastjson_decode

if TYPE_CHECKING:
    from collections.abc import Iterable

# Target schema: flat + nested + list. Shared by every scenario.
SCHEMA: dict[str, pl.DataType] = {
    "id": pl.String(),
    "score": pl.Float64(),
    "tags": pl.List(pl.String()),
    "nested": pl.Struct({"a": pl.Int64(), "b": pl.String()}),
}

# Default row-count sweep, used when --rows is not given.
SWEEP: tuple[int, ...] = (50_000, 200_000, 1_000_000)

DiagnosticsMode = Literal["off", "summary", "summary-id"]

# Scenario names, in display order.
SCENARIOS: tuple[str, ...] = (
    "all-valid",
    "with-malformed",
    "with-off-schema",
    "mixed",
)

DIAGNOSTIC_MODES: tuple[DiagnosticsMode, ...] = ("off", "summary", "summary-id")


def _random_word(rng: random.Random, *, min_len: int = 3, max_len: int = 8) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choices(string.ascii_lowercase, k=n))


def _valid_row(rng: random.Random) -> str:
    obj = {
        "id": _random_word(rng),
        "score": round(rng.uniform(0.0, 100.0), 4),
        "tags": [_random_word(rng) for _ in range(rng.randint(0, 4))],
        "nested": {"a": rng.randint(-1000, 1000), "b": _random_word(rng)},
    }
    return json.dumps(obj)


def _off_schema_row(rng: random.Random) -> str:
    """Valid JSON whose shape disagrees with the schema (coercion/null path)."""
    choice = rng.randint(0, 3)
    if choice == 0:
        # score is a string instead of a float.
        obj: dict[str, Any] = {
            "id": _random_word(rng),
            "score": _random_word(rng),
            "tags": [_random_word(rng)],
            "nested": {"a": rng.randint(0, 10), "b": _random_word(rng)},
        }
    elif choice == 1:
        # nested is a scalar instead of an object.
        obj = {
            "id": _random_word(rng),
            "score": rng.uniform(0.0, 1.0),
            "tags": [],
            "nested": rng.randint(0, 10),
        }
    elif choice == 2:
        # Top-level array rather than an object.
        return json.dumps([_random_word(rng) for _ in range(3)])
    else:
        # Extra fields plus a missing required one (id absent).
        obj = {
            "score": rng.uniform(0.0, 1.0),
            "tags": [_random_word(rng)],
            "nested": {"a": 1, "b": "x"},
            "extra": _random_word(rng),
        }
    return json.dumps(obj)


def _malformed_row(rng: random.Random) -> str:
    """Syntactically invalid JSON."""
    choice = rng.randint(0, 3)
    if choice == 0:
        return "not json at all"
    if choice == 1:
        return '{"id": "a", "score": '  # truncated
    if choice == 2:
        return '{"id": "a" "score": 1.0}'  # missing comma
    return "{" + _random_word(rng)  # unbalanced brace


def generate_raw(
    *,
    rows: int,
    malformed_frac: float,
    off_schema_frac: float,
    seed: int,
) -> pl.DataFrame:
    """Build a ``raw`` column of JSON strings for one scenario.

    ``malformed_frac`` of rows are syntactically invalid; ``off_schema_frac`` are
    valid-but-off-schema; the remainder are valid in-schema objects. Setting both
    fractions to ``0.0`` yields the all-valid scenario.
    """
    rng = random.Random(seed)
    raws: list[str] = []
    for _ in range(rows):
        r = rng.random()
        if r < malformed_frac:
            raws.append(_malformed_row(rng))
        elif r < malformed_frac + off_schema_frac:
            raws.append(_off_schema_row(rng))
        else:
            raws.append(_valid_row(rng))
    return pl.DataFrame({"row_id": [f"row_{i}" for i in range(rows)], "raw": raws})


def _scenario_fracs(
    scenario: str, *, malformed_frac: float, off_schema_frac: float
) -> tuple[float, float]:
    """Return (malformed_frac, off_schema_frac) for a named scenario."""
    if scenario == "all-valid":
        return 0.0, 0.0
    if scenario == "with-malformed":
        return malformed_frac, 0.0
    if scenario == "with-off-schema":
        return 0.0, off_schema_frac
    if scenario == "mixed":
        return malformed_frac, off_schema_frac
    raise ValueError(f"unknown scenario: {scenario}")


def _run_fastjson(df: pl.DataFrame, *, diagnostics: DiagnosticsMode) -> pl.DataFrame:
    diagnostics_mode: Literal["off", "summary"] = (
        "off" if diagnostics == "off" else "summary"
    )
    diagnostics_id = "row_id" if diagnostics == "summary-id" else None
    return df.with_columns(
        fastjson_decode(
            pl.col("raw"),
            schema=SCHEMA,
            on_error="null",
            coerce=True,
            diagnostics=diagnostics_mode,
            diagnostics_id=diagnostics_id,
        ).alias("parsed")
    )


@dataclass(frozen=True)
class Result:
    scenario: str
    diagnostics: DiagnosticsMode
    rows: int
    median_ms: float

    @property
    def rows_per_sec(self) -> float:
        if self.median_ms <= 0:
            return float("inf")
        return self.rows / (self.median_ms / 1000.0)


def _time_diagnostics(
    df: pl.DataFrame, *, repeats: int, diagnostics: DiagnosticsMode
) -> float:
    """Median wall-clock (ms) for one diagnostics mode.

    Each run fully materializes the decoded column so timing reflects real
    decode work rather than lazy-plan construction.
    """
    _run_fastjson(df, diagnostics=diagnostics)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = _run_fastjson(df, diagnostics=diagnostics)
        _ = out["parsed"].len()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_benchmark(
    *,
    row_counts: list[int],
    repeats: int,
    seed: int,
    malformed_frac: float,
    off_schema_frac: float,
    diagnostics_modes: list[DiagnosticsMode],
) -> list[Result]:
    results: list[Result] = []
    for scenario in SCENARIOS:
        m_frac, o_frac = _scenario_fracs(
            scenario, malformed_frac=malformed_frac, off_schema_frac=off_schema_frac
        )
        for rows in row_counts:
            df = generate_raw(
                rows=rows,
                malformed_frac=m_frac,
                off_schema_frac=o_frac,
                seed=seed,
            )
            for diagnostics in diagnostics_modes:
                results.append(
                    Result(
                        scenario=scenario,
                        diagnostics=diagnostics,
                        rows=rows,
                        median_ms=_time_diagnostics(
                            df, repeats=repeats, diagnostics=diagnostics
                        ),
                    )
                )
    return results


def _print_table(results: Iterable[Result]) -> None:
    rows = list(results)
    baseline = {
        (res.scenario, res.rows): res.median_ms
        for res in rows
        if res.diagnostics == "off"
    }
    header = (
        f"{'scenario':<18}"
        f"{'diagnostics':<13}"
        f"{'rows':>12}"
        f"{'median ms':>14}"
        f"{'rows/sec':>16}"
        f"{'vs off':>10}"
    )
    print(header)
    print("-" * len(header))
    for res in rows:
        base = baseline.get((res.scenario, res.rows))
        overhead = ""
        if base is not None and res.diagnostics != "off":
            overhead = f"{((res.median_ms / base) - 1.0) * 100.0:+.1f}%"
        print(
            f"{res.scenario:<18}"
            f"{res.diagnostics:<13}"
            f"{res.rows:>12,}"
            f"{res.median_ms:>14.2f}"
            f"{res.rows_per_sec:>16,.0f}"
            f"{overhead:>10}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="single row count (disables the default SWEEP)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed repeats per cell (median)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--malformed-frac",
        type=float,
        default=0.05,
        help="fraction of syntactically invalid rows in malformed scenarios",
    )
    parser.add_argument(
        "--off-schema-frac",
        type=float,
        default=0.05,
        help="fraction of valid-but-off-schema rows in off-schema scenarios",
    )
    parser.add_argument(
        "--diagnostics",
        choices=("off", "summary", "summary-id", "all"),
        default="off",
        help="diagnostics mode(s) to benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 0.0 <= args.malformed_frac < 1.0:
        raise SystemExit("--malformed-frac must be in [0.0, 1.0)")
    if not 0.0 <= args.off_schema_frac < 1.0:
        raise SystemExit("--off-schema-frac must be in [0.0, 1.0)")
    if args.malformed_frac + args.off_schema_frac >= 1.0:
        raise SystemExit("--malformed-frac + --off-schema-frac must be < 1.0")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    row_counts = [args.rows] if args.rows is not None else list(SWEEP)
    if any(r < 1 for r in row_counts):
        raise SystemExit("--rows must be >= 1")
    if args.diagnostics == "all":
        diagnostics_modes: list[DiagnosticsMode] = list(DIAGNOSTIC_MODES)
    else:
        diagnostics_modes = [cast("DiagnosticsMode", args.diagnostics)]

    print(
        f"fastjson_decode self-benchmark\n"
        f"schema: id:str, score:f64, tags:list[str], nested:{{a:i64, b:str}}\n"
        f"row_counts={row_counts} repeats={args.repeats} seed={args.seed} "
        f"diagnostics={diagnostics_modes} "
        f"malformed_frac={args.malformed_frac:.3f} "
        f"off_schema_frac={args.off_schema_frac:.3f}\n"
    )

    results = run_benchmark(
        row_counts=row_counts,
        repeats=args.repeats,
        seed=args.seed,
        malformed_frac=args.malformed_frac,
        off_schema_frac=args.off_schema_frac,
        diagnostics_modes=diagnostics_modes,
    )
    _print_table(results)


if __name__ == "__main__":
    main()

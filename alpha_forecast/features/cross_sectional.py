"""Cross-sectional (sector-relative) momentum features for a universe of assets.

Single-asset features only describe an asset against its own history. For a
portfolio the question is *which* asset to favour, so these features describe
each asset's momentum relative to the rest of the universe and to its sector:

    * ``xs_rank_{w}`` -- percentile rank of ``w``-day momentum across the universe
    * ``xs_z_{w}``    -- cross-sectional z-score of ``w``-day momentum
    * ``sec_rel_{w}`` -- momentum minus the mean momentum of the asset's sector
                         peers (leave-one-out); assets with no sector peer are
                         compared with the rest of the universe instead

Momentum is the ``w``-day log return up to and including day ``t``, so every
value only uses prices observable at the close of ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (21, 63, 126)


def _leave_one_out_mean(mom: pd.DataFrame) -> pd.DataFrame:
    """Mean of the *other* columns' values on each date (NaN when there are none)."""
    total = mom.sum(axis=1, min_count=1)
    others = mom.notna().sum(axis=1) - 1
    peer_sum = mom.rsub(total, axis=0)
    return peer_sum.div(others.where(others > 0), axis=0)


def _peer_mean(mom: pd.DataFrame, sectors: Mapping[str, str] | None) -> pd.DataFrame:
    universe = _leave_one_out_mean(mom)
    if not sectors:
        return universe
    # Tickers are matched case-insensitively so "aapl" and "AAPL" share a sector.
    by_symbol = {str(k).upper(): v for k, v in sectors.items()}
    groups: dict[str, list[str]] = {}
    for symbol in mom.columns:
        key = by_symbol.get(str(symbol).upper(), f"__solo_{symbol}")
        groups.setdefault(key, []).append(symbol)
    sector = pd.concat([_leave_one_out_mean(mom[cols]) for cols in groups.values()], axis=1)
    sector = sector[mom.columns]
    return sector.where(sector.notna(), universe)


def cross_sectional_momentum(
    closes: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    sectors: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute cross-sectional momentum features for every asset in ``closes``.

    Parameters
    ----------
    closes:
        Close prices, one column per symbol, indexed by date. Gaps (e.g. an
        equity holiday in a universe that also holds crypto) are forward
        filled so every window spans the same dates; a symbol is left out of
        the cross-section only before its first price.
    windows:
        Momentum look-back windows in trading days.
    sectors:
        Optional ``{symbol: sector}`` map (case-insensitive). Symbols missing
        from it, or alone in their sector, are compared with the whole universe.

    Returns ``{symbol: DataFrame}`` with one row per date of ``closes``.
    """
    if closes.shape[1] < 2:
        raise ValueError("Cross-sectional features need at least two symbols")

    log_close = np.log(closes.astype(float).ffill())
    per_window: dict[str, pd.DataFrame] = {}
    for w in windows:
        mom = log_close.diff(w)
        mean = mom.mean(axis=1)
        std = mom.std(axis=1, ddof=0)
        per_window[f"xs_rank_{w}"] = mom.rank(axis=1, pct=True)
        # A flat cross-section (zero dispersion) gives every asset a z-score of 0.
        z = mom.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
        per_window[f"xs_z_{w}"] = z.fillna(0.0).where(mom.notna())
        per_window[f"sec_rel_{w}"] = mom - _peer_mean(mom, sectors)

    return {
        symbol: pd.DataFrame({name: frame[symbol] for name, frame in per_window.items()})
        for symbol in closes.columns
    }

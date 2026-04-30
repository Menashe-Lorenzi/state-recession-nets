"""Shared plotting helpers."""
from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes


def shade_recessions(ax: Axes, usrec: pd.Series, **kwargs) -> None:
    """Shade NBER recession periods as vertical bands on the given axis.

    ``usrec`` is the monthly 0/1 series indexed by month-end.
    """
    kwargs.setdefault("color", "gray")
    kwargs.setdefault("alpha", 0.25)
    kwargs.setdefault("linewidth", 0)

    in_rec = False
    start = None
    # Iterate in chronological order; close the last band at the final date.
    for date, flag in usrec.items():
        if flag == 1 and not in_rec:
            start = date
            in_rec = True
        elif flag == 0 and in_rec:
            ax.axvspan(start, date, **kwargs)
            in_rec = False
    if in_rec:
        ax.axvspan(start, usrec.index[-1], **kwargs)

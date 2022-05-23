"""Functions to change frequency of a pandas dataframe."""

from ..tools import stamps
from pandas.core.frame import NDFrame
import pandas as pd


def _general(fr: NDFrame, freq: str = "MS", is_summable: bool = True):
    """Change frequency of a Series or DataFrame, depending on the type of data it
    contains."""

    # TODO: Make sure result has correct dtype (currently float -> object sometimes)
    # TODO: Add tests with multiindex columns

    # Some resampling labels are right-bound by default. Change to make left-bound.
    if freq in ["M", "A", "Q"]:
        freq += "S"
    if freq not in stamps.FREQUENCIES:
        raise ValueError(
            f"Parameter ``freq`` must be one of {','.join(stamps.FREQUENCIES)}; got {freq}."
        )

    # Empty frame.
    if len(fr) == 0:
        return fr.resample(freq).mean()  # empty frame.

    up_or_down = stamps.freq_up_or_down(fr.index.freq, freq)

    # Nothing more needed; portfolio already in desired frequency.
    if up_or_down == 0:
        return fr

    # Must downsample.
    elif up_or_down == -1:
        if is_summable:
            # Downsampling is easiest for summable frames: simply sum child values.
            fr2 = fr.resample(freq).sum()
            # Discard rows in new frame that are only partially present in original.
            fr2 = fr2[
                (fr2.index >= fr.index[0])
                & (fr2.index.ts_right <= fr.index.ts_right[-1])
            ]
            # Return if any values found
            if not len(fr2):
                raise ValueError("There are no 'full' time periods at this frequency.")
            return fr2
        else:
            # For averagable frames: first make summable.
            summable = fr.mul(fr.index.duration, axis=0)
            summable2 = _general(summable, freq, True)
            fr2 = summable2.div(summable2.index.duration, axis=0)
            return fr2 if isinstance(fr2, pd.DataFrame) else fr2.rename(fr.name)

    # Must upsample.
    else:
        if not is_summable:
            # Upsampling is easiest for averagable frames: simply duplicate parent value.
            # We cannot simply `.resample()`, because in that case the final value is not
            # duplicated. We add a dummy value, which we eventually remove again.

            # (original code to add additional row, does not work if SERIES and unit-aware. Maybe with future release of pint_pandas?)
            # fr = fr.copy()
            # if isinstance(fr, pd.Series):
            #     fr.loc[fr.index.ts_right[-1]] = None
            # else:
            #     fr.loc[fr.index.ts_right[-1], :] = None

            if isinstance(fr, pd.Series):
                # Workaround: turn into dataframe, change frequency, and turn back into series.
                return _general(pd.DataFrame(fr), freq, is_summable).iloc[:, 0]

            fr = fr.copy()  # don't change incoming dataframe
            # first, add additional row...
            fr.loc[fr.index.ts_right[-1], :] = None
            # ... then do upsampling ...
            fr2 = fr.resample(freq).asfreq().ffill()  # duplicate value
            # ... and then remove final row.
            return fr2.iloc[:-1]
        else:
            # For summable frames: first make averagable.
            avgable = fr.div(fr.index.duration, axis=0)
            avgable2 = _general(avgable, freq, False)
            fr2 = avgable2.mul(avgable2.index.duration, axis=0)
            return fr2 if isinstance(fr2, pd.DataFrame) else fr2.rename(fr.name)


def summable(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-summable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-summable' quantity is one that can be summed to get to an aggregate
    value, like revenue [Eur] or energy [MWh]. Prices [Eur/MWh] and powers [MW]
    are not time-summable.
    """
    return _general(fr, freq, True)


def averagable(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-averagable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-averagable' quantity is one that can be averaged to an aggregate value,
    like power [MW]. When downsampling, the values are weighted with their duration.
    """
    return _general(fr, freq, False)

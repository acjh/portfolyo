from typing import Iterable
from portfolyo import dev, testing
from portfolyo.tools import frames
from numpy import nan
import portfolyo as pf
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_aware", [True, False])
@pytest.mark.parametrize("bound", ["right", "left"])
@pytest.mark.parametrize(
    ("values", "i", "dst_transition"),
    [
        (
            np.random.random(96),
            pd.date_range(
                "2020-03-01",
                "2020-03-02",
                freq="15T",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            False,
        ),
        (  # Check for days incl WT->ST.
            np.random.random(92),
            pd.date_range(
                "2020-03-29",
                "2020-03-30",
                freq="15T",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            True,
        ),
        (  # Check for days incl ST->WT.
            np.random.random(100),
            pd.date_range(
                "2020-10-25",
                "2020-10-26",
                freq="15T",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            True,
        ),
    ],
)
def test_settsindex_15T(
    values: Iterable,
    i: pd.DatetimeIndex,
    bound: str,
    tz_aware: bool,
    dst_transition: bool,
    tz: str,
    series_or_df: str,
):
    """Test if index of dataframes and series is correctly standardized for quarterhour
    timeseries with/without DST."""
    # Prepare index.
    i = i.tz_convert(tz).rename("ts_left")
    if not tz_aware:
        i = i.tz_localize(None)
    i.freq = pd.infer_freq(i)

    # Prepare expected output frame.
    if series_or_df == "df":
        expected = pd.DataFrame({"a": values}, i)
    else:
        expected = pd.Series(values, i)

    # Prepare expected input frame.
    i = i.rename("the_time_stamp")
    if bound == "right":
        i = i + pd.Timedelta("15T")

    # Prepare input frame, calculate result and compared to expected.
    continuous = not dst_transition or tz != "Europe/Berlin" or tz_aware
    kwargs = {"continuous": continuous, "bound": bound}
    if series_or_df == "df":
        # 1: Using expected frame: should stay the same.
        result = frames.set_ts_index(expected, continuous=continuous)
        pd.testing.assert_frame_equal(result, expected)
        # 2: Dataframe with index.
        result = frames.set_ts_index(pd.DataFrame({"a": values}, i), **kwargs)
        pd.testing.assert_frame_equal(result, expected)
        # 3: Dataframe with column that must become index.
        result = frames.set_ts_index(pd.DataFrame({"a": values, "t": i}), "t", **kwargs)
        pd.testing.assert_frame_equal(result, expected)
    else:
        # 1: Using expected frame: should stay the same.
        result = frames.set_ts_index(expected, continuous=continuous)
        pd.testing.assert_series_equal(result, expected)
        # 2: Series.
        result = frames.set_ts_index(pd.Series(values, i), **kwargs)
        pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("removesome", [0, 1, 2])  # 0=none, 1=from end, 2=from middle
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq", [*pf.FREQUENCIES, "Q", "30T", "M", "AS-FEB"])
def test_settsindex_2(freq, tz, removesome, series_or_df: str):
    """Test raising errors on incorrect frequencies or indices with gaps."""

    must_raise = False

    # Get index.
    while True:
        i = dev.get_index(freq, tz)
        if len(i) > 8:
            break
    # If no timezone specified and below-daily values, the created index will have too few/many datapoints.
    if not tz and pf.freq_up_or_down(freq, "D") > 1:
        return  # don't check this edge case

    if tz == "Asia/Kolkata" and pf.freq_shortest(freq, "H") == "H":
        # Kolkata and Berlin timezone only share 15T-boundaries. Therefore, any other
        # frequency should raise an error.
        must_raise = True

    for _ in range(1, 3):  # remove 1 or 2 values
        if removesome == 1:
            i = i.delete(np.random.choice([0, len(i) - 1]))
        elif removesome == 2:
            i = i.delete(np.random.randint(1, len(i) - 1))

    # Add values.
    if series_or_df == "series":
        fr = dev.get_series(i)
    else:
        fr = dev.get_dataframe(i)

    # See if error is raised.
    if removesome == 2 or freq not in pf.FREQUENCIES or must_raise:
        with pytest.raises(ValueError):
            frames.set_ts_index(fr, continuous=True)
        return

    assert frames.set_ts_index(fr, tz_in="Europe/Berlin").index.freq == freq


@pytest.mark.parametrize(
    ("values", "maxgap", "gapvalues"),
    [
        ([1, 2, 3, 4, 25, 7, 8], 1, []),
        ([1, 2, 3, 4, nan, 7, 8], 1, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 2, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 3, [5.5]),
        ([3, 2, 1, nan, nan, 7, 8], 1, [nan, nan]),
        ([3, 2, 1, nan, nan, 7, 8], 2, [3, 5]),
        ([3, 2, 1, nan, nan, 7, 8], 3, [3, 5]),
        ([nan, 2, 1, nan, nan, 7, nan], 1, [nan, nan, nan, nan]),
        ([nan, 2, 1, nan, nan, 7, nan], 2, [nan, 3, 5, nan]),
    ],
)
@pytest.mark.parametrize(
    ("index", "tol"),
    [
        (range(7), 0),
        (range(-3, 4), 0),
        (pd.date_range("2020", periods=7, freq="D"), 0),
        (pd.date_range("2020", periods=7, freq="M", tz="Europe/Berlin"), 0.04),
    ],
)
def test_fill_gaps(values, index, maxgap, gapvalues, tol):
    """Test if gaps are correctly interpolated."""
    # Test as Series.
    s = pd.Series(values, index)
    s_new = frames.fill_gaps(s, maxgap)
    s[s.isna()] = gapvalues
    pd.testing.assert_series_equal(s_new, s, rtol=tol)
    # Test as DataFrame.
    df = pd.DataFrame({"a": values}, index)
    df_new = frames.fill_gaps(df, maxgap)
    df[df.isna()] = gapvalues
    pd.testing.assert_frame_equal(df_new, df, rtol=tol)


@pytest.mark.parametrize(
    ("df_columns", "header", "expected_columns"),
    [
        (["A"], "B", [("B", "A")]),
        (["A1", "A2"], "B", [("B", "A1"), ("B", "A2")]),
        (pd.MultiIndex.from_tuples([("B", "A")]), "C", [("C", "B", "A")]),
        (
            pd.MultiIndex.from_product([["B"], ["A1", "A2"]]),
            "C",
            [("C", "B", "A1"), ("C", "B", "A2")],
        ),
        (
            pd.MultiIndex.from_tuples([("B1", "A1"), ("B2", "A2")]),
            "C",
            [("C", "B1", "A1"), ("C", "B2", "A2")],
        ),
    ],
)
def test_addheader_tocolumns(df_columns, header, expected_columns):
    """Test if header can be added to the columns of a dataframe."""
    i = dev.get_index()
    df_in = pd.DataFrame(np.random.rand(len(i), len(df_columns)), i, df_columns)
    result_columns = frames.add_header(df_in, header).columns.to_list()
    assert np.array_equal(result_columns, expected_columns)


# TODO: put in ... fixture (?)
test_index_D = dev.get_index("D")
test_index_D_deconstructed = test_index_D.map(lambda ts: (ts.year, ts.month, ts.day))
test_index_H = dev.get_index("H")
test_index_H_deconstructed = test_index_H.map(lambda ts: (ts.year, ts.month, ts.day))


@pytest.mark.parametrize(
    ("df_index", "header", "expected_index"),
    [
        (test_index_D, "test", [("test", i) for i in test_index_D]),
        (
            test_index_D_deconstructed,
            "test",
            [("test", *i) for i in test_index_D_deconstructed],
        ),
        (test_index_H, "test", [("test", i) for i in test_index_H]),
        (
            test_index_H_deconstructed,
            "test",
            [("test", *i) for i in test_index_H_deconstructed],
        ),
    ],
)
def test_addheader_torows(df_index, header, expected_index):
    """Test if header can be added to the rows of a dataframe."""
    df_in = pd.DataFrame(np.random.rand(len(df_index), 2), df_index, ["A", "B"])
    result_index = frames.add_header(df_in, header, axis=0).index.to_list()
    assert np.array_equal(result_index, expected_index)


# TODO: put in ... fixture (?)
test_values = np.random.rand(len(test_index_D), 10)
test_df_1 = pd.DataFrame(test_values[:, :2], test_index_D, ["A", "B"])
test_df_2 = pd.DataFrame(test_values[:, 2], test_index_D, ["C"])
expect_concat_12 = pd.DataFrame(test_values[:, :3], test_index_D, ["A", "B", "C"])
test_df_3 = pd.DataFrame(test_values[:, 2], test_index_D, pd.Index([("D", "C")]))
expect_concat_13 = pd.DataFrame(
    test_values[:, :3], test_index_D, pd.Index([("A", ""), ("B", ""), ("D", "C")])
)


@pytest.mark.parametrize(
    ("dfs", "axis", "expected"),
    [
        ([test_df_1, test_df_2], 1, expect_concat_12),
        ([test_df_1, test_df_3], 1, expect_concat_13),
    ],
)
def test_concat(dfs, axis, expected):
    """Test if concatenation works as expected."""
    result = frames.concat(dfs, axis)
    testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("weightsas", ["none", "list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasseries1(weightsas, axis):
    """Test if weighted average of a series is correctly calculated."""
    values = pd.Series([100, 200, 300, -150])
    weights = [10, 10, 10, 20]
    if weightsas == "none":
        weights = None
        expected_result = 112.5
    elif weightsas == "list":
        expected_result = 60
    elif weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
        expected_result = 110
    assert np.isclose(frames.wavg(values, weights, axis), expected_result)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasseries2(weightsas, axis):
    """Test if weighted average of a series is correctly calculated."""
    values = pd.Series([100, 200, 300, -150])
    weights = [10, 0, 10, 20]
    if weightsas == "list":
        expected_result = 25
    elif weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
        expected_result = 62.5
    assert np.isclose(frames.wavg(values, weights, axis), expected_result)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasseries_na(weightsas, axis):
    """Test if weighted average of a series is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    values = pd.Series([100, 200, 300, -150])
    weights = [0, 0, 0, 0]
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    assert np.isnan(frames.wavg(values, weights, axis))


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasseries_0weights(weightsas, axis):
    """Test if weighted average of a series is correctly calculated,
    when all weights are 0 and all values are equal."""
    values = pd.Series([100, 100, 100, 100])
    weights = [0, 0, 0, 0]
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    assert frames.wavg(values, weights, axis) == 100


@pytest.mark.parametrize("weightsas", ["none", "list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe1(weightsas, axis):
    """Test if weighted average of a dataframe is correctly calculated."""
    values = pd.DataFrame({"a": [100, 200, 300, -150], "b": [100, -200, 300, -150]})
    if weightsas == "none":
        weights = None
        if axis == 0:
            expected_result = pd.Series({"a": 112.5, "b": 12.5})
        else:
            expected_result = pd.Series([100, 0, 300, -150])
    if weightsas == "list":
        if axis == 0:
            weights = [10, 10, 10, 20]
            expected_result = pd.Series({"a": 60, "b": -20})
        else:
            weights = [10, 30]
            expected_result = pd.Series([100, -100, 300, -150])
    if weightsas == "series":
        if axis == 0:
            weights = pd.Series([10, 10, 10, 20], index=[3, 2, 1, 0])
            expected_result = pd.Series({"a": 110, "b": 30})
        else:
            weights = pd.Series({"b": 30, "a": 10})
            expected_result = pd.Series([100, -100, 300, -150])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10, 10, 10, 20], "b": [10, 10, 30, 0]})
        if axis == 0:
            expected_result = pd.Series({"a": 60, "b": 160})
        else:
            expected_result = pd.Series([100, 0, 300, -150])
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe2(weightsas, axis):
    """Test if weighted average of a dataframe is correctly calculated."""
    values = pd.DataFrame({"a": [100, 200, 200, -150], "b": [100, -200, 300, -150]})
    if weightsas == "list":
        if axis == 0:
            weights = [10, 10, 0, 20]
            expected_result = pd.Series({"a": 0, "b": -100})
        else:
            weights = [10, 0]
            expected_result = pd.Series([100, 200, 200, -150])
    if weightsas == "series":
        if axis == 0:
            weights = pd.Series([10, 10, 0, 20], index=[3, 2, 1, 0])
            expected_result = pd.Series({"a": 62.5, "b": 87.5})
        else:
            weights = pd.Series({"b": 0, "a": 10})
            expected_result = pd.Series([100, 200, 200, -150])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10, 10, 0, 20], "b": [10, 10, 30, 0]})
        if axis == 0:
            expected_result = pd.Series({"a": 0, "b": 160})
        else:
            expected_result = pd.Series([100, 0, 300, -150])
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe_na(weightsas, axis):
    """Test if weighted average of a dataframe is correctly is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    values = pd.DataFrame({"a": [130, 200, 200, -160], "b": [100, -200, 300, -150]})
    if axis == 0:
        weights = [0, 0, 0, 0]
        expected_result = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected_result = pd.Series([np.nan, np.nan, np.nan, np.nan])

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, index=[3, 2, 1, 0])
        else:
            weights = pd.Series(weights, index=["a", "b"])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]})
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe_0weights(weightsas, axis):
    """Test if weighted average of a dataframe is correctly is correctly identified as error,
    when all weights are 0. Some averages are calculated from identical values and should
    result in that value."""
    values = pd.DataFrame({"a": [100, 200, 200, -150], "b": [100, -200, 300, -150]})
    if axis == 0:
        weights = [0, 0, 0, 0]
        expected_result = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected_result = pd.Series([100, np.nan, np.nan, -150])

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, index=[3, 2, 1, 0])
        else:
            weights = pd.Series(weights, index=["a", "b"])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]})
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


vals1 = np.array([1, 2.0, -4.1234, 0])
vals2 = np.array([1, 2.0, -4.1234, 0.5])


@pytest.mark.parametrize(
    ("s1", "s2", "expected_result"),
    [
        (pd.Series(vals1), pd.Series(vals1), True),
        (pd.Series(vals1), pd.Series(vals2), False),
        (pd.Series(vals1), pd.Series(vals1, dtype="pint[MW]"), False),
        (pd.Series(vals1).astype("pint[MW]"), pd.Series(vals1, dtype="pint[MW]"), True),
        (
            pd.Series(vals1 * 1000).astype("pint[kW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            True,
        ),
        (
            pd.Series(vals1 * 1000).astype("pint[MW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            False,
        ),
    ],
)
def test_series_allclose(s1, s2, expected_result):
    """Test if series can be correctly compared, even if they have a unit."""
    assert frames.series_allclose(s1, s2) == expected_result

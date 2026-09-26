"""Exercise the installed extension, including stable ABI and WebAssembly wheels."""

from datetime import date
from math import isclose

import pyxirr


def expect_error(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


dates = [date(2020, 1, 1), date(2021, 1, 1)]
amounts = [-100, 110]
assert pyxirr.xnpv(0, dates, amounts) == 10
assert pyxirr.xnpv([0, 0], dates, amounts) == [10, 10]
assert pyxirr.xnpv([], dates, amounts) == []
assert pyxirr.npv([[]], [100]) == [[]]
assert pyxirr.npv([[0, 0, 0], [0, 0, 0]], [100]) == [[100] * 3] * 2
assert pyxirr.is_conventional_cash_flow([-100, 0, 110])
assert isclose(
    pyxirr.xnpv(0.1, dates, amounts, day_count=pyxirr.DayCount.ACT_365F),
    pyxirr.xnpv(0.1, dates, amounts, day_count="Actual/365F"),
)
assert pyxirr.xnpv(0.1, [], [], silent=True) is None
assert pyxirr.xnpv([0.1, 0.2], [], [], silent=True) == [None, None]
for rates in (0.1, []):
    expect_error(pyxirr.InvalidPaymentsError, pyxirr.xnpv, rates, [], [])
expect_error(ValueError, pyxirr.npv, [[], [[], []]], [100])
expect_error(TypeError, pyxirr.npv, ["invalid"], [100])
cyclic = []
cyclic.append(cyclic)
expect_error(ValueError, pyxirr.npv, cyclic, [100])
deep = 0.1
for _ in range(65):
    deep = [deep]
expect_error(RecursionError, pyxirr.npv, deep, [100])


class OverriddenDate(date):
    @property
    def month(self):
        return 0


assert pyxirr.xnpv(0, [OverriddenDate(2026, 1, 1)], [100]) == 100
for name in ("Timestamp", "NaTType"):
    subclass = type(name, (date,), {"__module__": "example"})
    assert pyxirr.xnpv(0, [subclass(2026, 1, 1)], [100]) == 100

try:
    import numpy as np
except ImportError:
    print("Installed wheel smoke passed without NumPy")
else:
    assert pyxirr.xnpv(0, np.array(dates, dtype="datetime64[D]"), amounts) == 10
    assert pyxirr.xnpv(0, dates, np.array(amounts)) == 10
    assert len(pyxirr.xnpv(np.array([]), dates, amounts)) == 0
    expect_error(pyxirr.InvalidPaymentsError, pyxirr.xnpv, np.array([]), [], [])
    for invalid in (
        np.datetime64("NaT", "D"),
        np.datetime64(2**32, "D"),
        np.datetime64((1 << 64) // 7 + 1, "W"),
    ):
        for date_input in ([invalid], np.array([invalid])):
            expect_error(ValueError, pyxirr.xnpv, 0, date_input, [100])
    early = np.datetime64(-(1 << 63) + 1, "ns")
    expected = pyxirr.xnpv(0.1, [date(1677, 9, 21), date(2020, 1, 1)], amounts)
    assert isclose(pyxirr.xnpv(0.1, [early, dates[0]], amounts), expected)
    early_dates = np.array([early, np.datetime64("2020-01-01", "ns")])
    assert isclose(pyxirr.xnpv(0.1, early_dates, amounts), expected)
    print("Installed wheel smoke passed with NumPy")

try:
    import pandas as pd
except ImportError:
    pass
else:
    for missing in ([pd.NaT], pd.DatetimeIndex([pd.NaT])):
        expect_error(ValueError, pyxirr.xnpv, 0, missing, [100])
    local_dates = pd.Series(
        [
            pd.Timestamp("2021-01-01 00:30:00+01:00"),
            pd.Timestamp("2022-01-01 12:00:00+01:00"),
        ]
    )
    expected = pyxirr.xnpv(0.1, [date(2021, 1, 1), date(2022, 1, 1)], amounts)
    assert isclose(pyxirr.xnpv(0.1, local_dates, amounts), expected)
    frame = pd.DataFrame({"date": local_dates, "amount": amounts})
    assert isclose(pyxirr.xnpv(0.1, frame), expected)
    print("Installed wheel smoke passed with pandas")

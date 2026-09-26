use std::str::FromStr;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::*,
};
use time::Date;

use crate::core::{DateLike, DayCount};

// time::Date::from_ordinal_date(1970, 1).unwrap().to_julian_day();
static UNIX_EPOCH_JULIAN_DAY: i32 = 2440588;

pub fn float_or_none(result: f64) -> Option<f64> {
    if result.is_nan() {
        None
    } else {
        Some(result)
    }
}

pub fn fallible_float_or_none<T>(result: Result<f64, T>, silent: bool) -> PyResult<Option<f64>>
where
    pyo3::PyErr: From<T>,
{
    match result {
        Err(e) => {
            if silent {
                Ok(None)
            } else {
                Err(e.into())
            }
        }
        Ok(v) => Ok(float_or_none(v)),
    }
}

#[derive(FromPyObject)]
pub enum PyDayCount {
    String(String),
    DayCount(DayCount),
}

impl TryInto<DayCount> for PyDayCount {
    type Error = PyErr;

    fn try_into(self) -> Result<DayCount, Self::Error> {
        match self {
            PyDayCount::String(s) => DayCount::of(&s),
            PyDayCount::DayCount(d) => Ok(d),
        }
    }
}

#[pymethods]
impl DayCount {
    #[staticmethod]
    fn of(value: &str) -> PyResult<Self> {
        DayCount::from_str(value).map_err(PyValueError::new_err)
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

fn date_from_unix_days(days: i128) -> PyResult<DateLike> {
    let julian_day = days
        .checked_add(i128::from(UNIX_EPOCH_JULIAN_DAY))
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| PyValueError::new_err("date is outside the supported range"))?;
    Date::from_julian_day(julian_day)
        .map(Into::into)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

enum NumpyDateUnit {
    Years(i128),
    Months(i128),
    Fixed {
        multiplier: i128,
        divisor: i128,
    },
}

impl NumpyDateUnit {
    fn from_dtype(dtype: &Bound<PyAny>) -> PyResult<Self> {
        let (unit, step): (String, u32) =
            dtype.py().import("numpy")?.getattr("datetime_data")?.call1((dtype,))?.extract()?;
        if step == 0 {
            return Err(PyValueError::new_err("datetime units must have a positive multiplier"));
        }
        let step = i128::from(step);
        let divisor = match unit.as_str() {
            "Y" => return Ok(Self::Years(step)),
            "M" => return Ok(Self::Months(step)),
            "W" => {
                return Ok(Self::Fixed {
                    multiplier: step * 7,
                    divisor: 1,
                })
            }
            "D" => 1,
            "h" => 24,
            "m" => 1_440,
            "s" => 86_400,
            "ms" => 86_400_000,
            "us" => 86_400_000_000,
            "ns" => 86_400_000_000_000,
            "ps" => 86_400_000_000_000_000,
            "fs" => 86_400_000_000_000_000_000,
            "as" => 86_400_000_000_000_000_000_000,
            _ => {
                return Err(PyValueError::new_err(
                    "NaT or unspecified datetime units are not valid dates",
                ))
            }
        };
        Ok(Self::Fixed {
            multiplier: step,
            divisor,
        })
    }

    fn convert(&self, count: i64) -> PyResult<DateLike> {
        if count == i64::MIN {
            return Err(PyValueError::new_err("NaT is not a valid date"));
        }
        // Widen before scaling. NumPy's int64 unit casts can overflow before date validation.
        let count = i128::from(count);
        let (year, month) = match *self {
            Self::Years(step) => (1970 + count * step, 1),
            Self::Months(step) => {
                let months = count * step;
                (1970 + months.div_euclid(12), (months.rem_euclid(12) + 1) as u8)
            }
            Self::Fixed {
                multiplier,
                divisor,
            } => {
                return date_from_unix_days((count * multiplier).div_euclid(divisor));
            }
        };
        let year = i32::try_from(year)
            .map_err(|_| PyValueError::new_err("date is outside the supported range"))?;
        let month =
            time::Month::try_from(month).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Date::from_calendar_date(year, month, 1)
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl TryFrom<&Bound<'_, PyDate>> for DateLike {
    type Error = PyErr;

    fn try_from(value: &Bound<'_, PyDate>) -> Result<Self, Self::Error> {
        #[cfg(feature = "abi")]
        let (year, month, day) = {
            let date_type = value.py().get_type::<PyDate>();
            let field = |name| date_type.getattr(name)?.call_method1("__get__", (value,));
            (
                field("year")?.extract::<i32>()?,
                field("month")?.extract::<u8>()?,
                field("day")?.extract::<u8>()?,
            )
        };
        #[cfg(not(feature = "abi"))]
        let (year, month, day) = (value.get_year(), value.get_month(), value.get_day());
        let month =
            time::Month::try_from(month).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Date::from_calendar_date(year, month, day)
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl<'py> FromPyObject<'_, 'py> for DateLike {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        let obj_type = obj.get_type();
        let type_name = obj_type.name()?;
        let type_name = type_name.to_cow()?;
        if type_name == "NaTType" && obj_type.module()?.to_cow()? == "pandas._libs.tslibs.nattype" {
            return Err(PyValueError::new_err("NaT is not a valid date"));
        }
        if type_name == "Timestamp"
            && obj_type.module()?.to_cow()? == "pandas._libs.tslibs.timestamps"
        {
            return obj
                .call_method1(intern!(py, "to_pydatetime"), (false,))?
                .cast::<PyDate>()?
                .try_into();
        }
        if let Ok(py_date) = obj.cast::<PyDate>() {
            return py_date.try_into();
        }

        if let Ok(py_string) = obj.cast::<PyString>() {
            return py_string
                .to_cow()?
                .parse::<DateLike>()
                .map_err(|e| PyValueError::new_err(e.to_string()));
        }

        match type_name.as_ref() {
            "datetime64" => {
                let unit = NumpyDateUnit::from_dtype(&obj.getattr("dtype")?)?;
                let count = obj
                    .call_method1(intern!(py, "astype"), (intern!(py, "int64"),))?
                    .extract::<i64>()?;
                unit.convert(count)
            }

            other => Err(PyTypeError::new_err(format!(
                "Type {other:?} is not understood. Expected: date"
            ))),
        }
    }
}

fn extract_iterable<'a, T>(values: &Bound<'a, PyAny>) -> PyResult<Vec<T>>
where
    T: FromPyObjectOwned<'a>,
{
    values.try_iter()?.map(|i| i.and_then(|j| j.extract().map_err(Into::into))).collect()
}

fn extract_date_series_from_numpy(series: &Bound<PyAny>) -> PyResult<Vec<DateLike>> {
    let py = series.py();
    let dtype = series.getattr("dtype")?;
    if dtype.getattr("kind")?.extract::<String>()? != "M" {
        return extract_iterable::<DateLike>(series);
    }
    let unit = NumpyDateUnit::from_dtype(&dtype)?;
    series
        .call_method1(intern!(py, "astype"), (intern!(py, "int64"),))?
        .cast::<PyArray1<i64>>()?
        .readonly()
        .as_slice()?
        .iter()
        .map(|&x| unit.convert(x))
        .collect()
}

pub fn extract_date_series(series: &Bound<PyAny>) -> PyResult<Vec<DateLike>> {
    match series.get_type().name()?.to_cow()?.as_ref() {
        "Series" => {
            let values = series.call_method0(intern!(series.py(), "to_numpy"))?;
            extract_date_series_from_numpy(&values)
        }
        "ndarray" => extract_date_series_from_numpy(series),
        _ => extract_iterable::<DateLike>(series),
    }
}

fn extract_amount_series_from_numpy(series: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    let py = series.py();
    Ok(series
        .call_method1(intern!(py, "astype"), (intern!(py, "float64"),))?
        .extract::<numpy::PyReadonlyArray1<f64>>()?
        .to_vec()?)
}

fn extract_records(data: &Bound<PyAny>) -> PyResult<(Vec<DateLike>, Vec<f64>)> {
    let capacity = data.len().unwrap_or(12); // pre-allocate vec
    let mut dates: Vec<DateLike> = Vec::with_capacity(capacity);
    let mut amounts: Vec<f64> = Vec::with_capacity(capacity);

    for obj in data.try_iter()? {
        let obj = obj?;
        // get_item() uses different ffi calls for different objects
        // PyTuple.get_item (ffi::PyTuple_GetItem) is faster than PyAny.get_item (ffi::PyObject_GetItem)
        let tup = if let Ok(py_tuple) = obj.cast::<PyTuple>() {
            (py_tuple.get_item(0)?, py_tuple.get_item(1)?)
        } else if let Ok(py_list) = obj.cast::<PyList>() {
            (py_list.get_item(0)?, py_list.get_item(1)?)
        } else {
            (obj.get_item(0)?, obj.get_item(1)?)
        };

        dates.push(tup.0.extract::<DateLike>()?);
        amounts.push(tup.1.extract::<f64>()?);
    }

    Ok((dates, amounts))
}

pub struct AmountArray(Vec<f64>);

impl AmountArray {
    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

impl<'s> FromPyObject<'_, 's> for AmountArray {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 's, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        extract_amount_series(obj).map(AmountArray)
    }
}

impl std::ops::Deref for AmountArray {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        self.0.as_ref()
    }
}

pub fn extract_amount_series(series: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    match series.get_type().name()?.to_cow()?.as_ref() {
        "Series" => {
            let values = series.getattr(intern!(series.py(), "values"))?;
            extract_amount_series_from_numpy(&values)
        }
        "ndarray" => extract_amount_series_from_numpy(series),
        _ => extract_iterable::<f64>(series),
    }
}

pub fn extract_payments(
    dates: &Bound<PyAny>,
    amounts: Option<&Bound<PyAny>>,
) -> PyResult<(Vec<DateLike>, Vec<f64>)> {
    if amounts.is_some() {
        return Ok((extract_date_series(dates)?, extract_amount_series(amounts.unwrap())?));
    };

    if let Ok(py_dict) = dates.cast::<PyDict>() {
        return Ok((
            extract_iterable::<DateLike>(py_dict.keys().as_any())?,
            extract_iterable::<f64>(py_dict.values().as_any())?,
        ));
    }

    let py = dates.py();

    match dates.get_type().name()?.to_cow()?.as_ref() {
        "DataFrame" => {
            let frame = dates;
            let columns = frame.getattr(intern!(py, "columns"))?;
            Ok((
                extract_date_series(&frame.get_item(columns.get_item(0)?)?)?,
                extract_amount_series(&frame.get_item(columns.get_item(1)?)?)?,
            ))
        }
        "Series" => {
            let index = &dates.getattr(intern!(py, "index"))?;

            if index.get_type().name()?.ne("DatetimeIndex") {
                return Err(PyTypeError::new_err("Expected Series with DatetimeIndex"));
            }

            Ok((extract_date_series(index)?, extract_amount_series(dates)?))
        }
        "ndarray" => {
            let array = dates;
            Ok((
                extract_date_series(&array.get_item(0)?)?,
                extract_amount_series(&array.get_item(1)?)?,
            ))
        }
        _ => extract_records(dates),
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{ffi::c_str, prelude::*, types::PyDict};
    use rstest::rstest;
    use time::{Date, Month};

    use crate::core::DateLike;

    fn get_locals<'p>(py: &'p Python) -> Bound<'p, PyDict> {
        py.eval(c_str!("{ 'np': __import__('numpy') }"), None, None)
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_extract_from_numpy_datetime_array() {
        Python::attach(|py| {
            let locals = &get_locals(&py);
            let data = py
                .eval(
                    c_str!("np.array(['2007-02-01', '2009-09-30'], dtype='datetime64[D]')"),
                    Some(locals),
                    None,
                )
                .unwrap();
            let dt: Vec<DateLike> = data.extract().unwrap();
            let exp: DateLike = Date::from_calendar_date(2007, Month::February, 1).unwrap().into();

            assert_eq!(dt[0], exp);
        })
    }

    #[rstest]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_extract_from_numpy_datetime() {
        Python::attach(|py| {
            let locals = &get_locals(&py);
            let data =
                py.eval(c_str!("np.datetime64('2007-02-01', '[D]')"), Some(locals), None).unwrap();
            let dt: DateLike = data.extract().unwrap();
            let exp: DateLike = Date::from_calendar_date(2007, Month::February, 1).unwrap().into();

            assert_eq!(dt, exp);
        })
    }

    #[rstest]
    #[case("np.datetime64('NaT')")]
    #[case("np.datetime64(2**32, 'D')")]
    #[case("np.datetime64('30000-01-01')")]
    #[case("np.datetime64((1 << 64) // 7 + 1, 'W')")]
    #[case("np.datetime64(1 << 62, 'Y')")]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_rejects_invalid_numpy_dates(#[case] input: &str) {
        Python::attach(|py| {
            let locals = &get_locals(&py);
            let input = std::ffi::CString::new(input).unwrap();
            let value = py.eval(&input, Some(locals), None).unwrap();
            let error = value.extract::<DateLike>().unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            let array = locals
                .get_item("np")
                .unwrap()
                .unwrap()
                .call_method1("array", (vec![value],))
                .unwrap();
            let error = super::extract_date_series(&array).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[rstest]
    fn test_date_subclass_uses_native_fields() {
        Python::attach(|py| {
            let value = py.eval(c_str!(
                "type('OverriddenDate', (__import__('datetime').date,), {'month': property(lambda self: 0)})(2026, 1, 1)"
            ), None, None).unwrap();
            let date = value.extract::<DateLike>().unwrap();
            assert_eq!(date, Date::from_calendar_date(2026, Month::January, 1).unwrap().into());
        });
    }

    #[rstest]
    #[case("Timestamp")]
    #[case("NaTType")]
    fn test_date_subclass_names_do_not_imply_pandas(#[case] name: &str) {
        Python::attach(|py| {
            let expression = std::ffi::CString::new(format!(
                "type('{name}', (__import__('datetime').date,), {{'__module__': 'example'}})(2026, 1, 1)"
            )).unwrap();
            let value = py.eval(&expression, None, None).unwrap();
            assert_eq!(value.extract::<DateLike>().unwrap(), "2026-01-01".parse().unwrap());
        });
    }

    #[rstest]
    #[case("np.datetime64(-(1 << 63) + 1, 'ns')", "1677-09-21")]
    #[case("np.datetime64(-1, 'ns')", "1969-12-31")]
    #[case("np.datetime64(-1, 'ps')", "1969-12-31")]
    #[case("np.datetime64(-1, 'fs')", "1969-12-31")]
    #[case("np.datetime64(-1, 'as')", "1969-12-31")]
    #[case("np.datetime64(-1, 'M')", "1969-12-01")]
    #[case("np.datetime64(-1, 'Y')", "1969-01-01")]
    #[case("np.datetime64(-1, 'W')", "1969-12-25")]
    #[case("np.datetime64(3, '3D')", "1970-01-10")]
    #[case("np.datetime64(-(1 << 63) + 1, '2147483647as')", "1342-05-04")]
    #[case("np.datetime64((1 << 63) - 1, '2147483647as')", "2597-08-29")]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_numpy_date_units(#[case] input: &str, #[case] expected: &str) {
        Python::attach(|py| {
            let locals = &get_locals(&py);
            let input = std::ffi::CString::new(input).unwrap();
            let value = py.eval(&input, Some(locals), None).unwrap();
            let expected = expected.parse::<DateLike>().unwrap();
            assert_eq!(value.extract::<DateLike>().unwrap(), expected);
            let array = locals
                .get_item("np")
                .unwrap()
                .unwrap()
                .call_method1("array", (vec![value],))
                .unwrap();
            assert_eq!(super::extract_date_series(&array).unwrap(), vec![expected]);
        });
    }

    #[rstest]
    #[case("[pd.NaT]")]
    #[case("pd.DatetimeIndex([pd.NaT])")]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_rejects_pandas_nat(#[case] input: &str) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("pd", py.import("pandas").unwrap()).unwrap();
            let input = std::ffi::CString::new(input).unwrap();
            let values = py.eval(&input, Some(&locals), None).unwrap();
            let error = super::extract_date_series(&values).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[rstest]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_rejects_pandas_timestamp_outside_calendar_range() {
        Python::attach(|py| {
            let numpy = py.import("numpy").unwrap();
            let pandas = py.import("pandas").unwrap();
            let date = numpy.call_method1("datetime64", ("30000-01-01", "s")).unwrap();
            match pandas.call_method1("Timestamp", (date,)) {
                Ok(value) => {
                    let error = value.extract::<DateLike>().unwrap_err();
                    assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                }
                // Older pandas versions reject this date during construction.
                Err(error) => assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py)),
            }
        });
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    #[cfg_attr(feature = "nonumpy", ignore)]
    fn test_timezone_aware_date_containers(#[case] dataframe: bool) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("pd", py.import("pandas").unwrap()).unwrap();
            let dates = py.eval(c_str!(
                "pd.Series([pd.Timestamp('2021-01-01 00:30:00+01:00'), pd.Timestamp('2022-01-01 12:00:00+01:00')])"
            ), Some(&locals), None).unwrap();
            let extracted = if dataframe {
                locals.set_item("dates", dates).unwrap();
                let frame = py
                    .eval(
                        c_str!("pd.DataFrame({'date': dates, 'amount': [-100, 110]})"),
                        Some(&locals),
                        None,
                    )
                    .unwrap();
                super::extract_payments(&frame, None).unwrap().0
            } else {
                super::extract_date_series(&dates).unwrap()
            };
            assert_eq!(
                extracted,
                vec!["2021-01-01".parse().unwrap(), "2022-01-01".parse().unwrap()]
            );
        });
    }
}

use std::{error::Error, fmt};

use ndarray::{ArrayD, ArrayViewD, Axis, CowArray, IxDyn};
use numpy::{npyffi, Element, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PY_ARRAY_API};
use pyo3::{
    exceptions::{PyRecursionError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyIterator, PyList, PySequence, PyString, PyTuple},
    IntoPyObjectExt,
};

use crate::conversions::float_or_none;

const MAX_ARRAY_DIMS: usize = 64;

/// An error returned when the payments do not contain both negative and positive payments.
#[derive(Debug)]
pub struct BroadcastingError(String);

impl BroadcastingError {
    pub fn new(shapes: &[&[usize]]) -> Self {
        Self(format!("{:?}", shapes))
    }
}

impl fmt::Display for BroadcastingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for BroadcastingError {}

pub fn broadcast_shapes(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    /* Discover the broadcast number of dimensions */
    let ndim = shapes.iter().map(|s| s.len()).max()?;
    let mut result = vec![0; ndim];

    /* Discover the broadcast shape in each dimension */
    for (i, cur) in result.iter_mut().enumerate() {
        *cur = 1;
        for s in shapes.iter() {
            /* This prepends 1 to shapes not already equal to ndim */
            if i + s.len() >= ndim {
                let k = i + s.len() - ndim;
                let tmp = s[k];
                if tmp == 1 {
                    continue;
                }
                if cur == &1 {
                    *cur = tmp;
                } else if cur != &tmp {
                    return None;
                }
            }
        }
    }

    Some(result)
}

#[macro_export]
macro_rules! broadcast_together {
    ($($a:expr),*) => {
        {
            let _a = &[$($a.shape(),)*];

            match $crate::broadcasting::broadcast_shapes(_a) {
                Some(shape) => Ok(( $($a.broadcast(shape.clone()).unwrap(),)*)),
                None => Err(BroadcastingError::new(_a))
            }
        }
    };
}

pub fn pyiter_to_arrayd<'py, T>(pyiter: Bound<'py, PyIterator>) -> PyResult<ArrayD<T>>
where
    T: FromPyObjectOwned<'py>,
{
    let mut flat_list = Vec::new();
    let mut ancestors = vec![pyiter.as_ptr()];
    let dims = flatten_pyiter(pyiter, &mut flat_list, &mut ancestors)?;
    let arr = ArrayD::from_shape_vec(IxDyn(&dims), flat_list);
    arr.map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn arrayd_to_pylist<'py>(
    py: Python<'py>,
    array: ArrayViewD<'_, f64>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    if array.ndim() == 1 {
        for &x in array {
            list.append(float_or_none(x).into_py_any(py)?)?;
        }
    } else {
        for subarray in array.axis_iter(Axis(0)) {
            let sublist = arrayd_to_pylist(py, subarray)?;
            list.append(sublist)?;
        }
    }
    Ok(list)
}

fn flatten_pyiter<'p, T>(
    pyiter: Bound<'p, PyIterator>,
    flat_list: &mut Vec<T>,
    ancestors: &mut Vec<*mut pyo3::ffi::PyObject>,
) -> PyResult<Vec<usize>>
where
    T: FromPyObjectOwned<'p>,
{
    let mut count = 0;
    let mut child_shape = None;
    for item in pyiter {
        let item = item?;
        count += 1;
        let shape = match item.extract::<T>() {
            Ok(val) => {
                flat_list.push(val);
                Vec::new()
            }
            Err(_) => {
                if item.is_instance_of::<PyString>() {
                    return Err(PyTypeError::new_err("strings are not numeric array values"));
                }
                if ancestors.contains(&item.as_ptr()) {
                    return Err(PyValueError::new_err("cyclic arrays are not supported"));
                }
                if ancestors.len() >= MAX_ARRAY_DIMS {
                    return Err(PyRecursionError::new_err(
                        "nested arrays support at most 64 dimensions",
                    ));
                }
                let sublist = item.try_iter()?;
                ancestors.push(item.as_ptr());
                let shape = flatten_pyiter(sublist, flat_list, ancestors)?;
                ancestors.pop();
                shape
            }
        };
        match &child_shape {
            Some(expected) if expected != &shape => {
                return Err(PyValueError::new_err("array dimensions must have matching lengths"));
            }
            None => child_shape = Some(shape),
            _ => {}
        }
    }

    let mut shape = vec![count];
    shape.extend(child_shape.unwrap_or_default());
    Ok(shape)
}

pub enum Arg<'p, T> {
    Scalar(T),
    Array(CowArray<'p, T, IxDyn>),
    NumpyArray(Bound<'p, PyArrayDyn<T>>),
}

impl<'p, T> Arg<'p, T>
where
    T: Clone + numpy::Element,
{
    pub fn into_arrayd(self) -> CowArray<'p, T, IxDyn> {
        self.into()
    }
}

fn is_numpy_available() -> bool {
    Python::attach(|py| py.import("numpy").is_ok())
}

fn pyarray_cast<'p, U: Element>(ob: &Bound<'p, PyAny>) -> PyResult<Bound<'p, PyArrayDyn<U>>> {
    let ptr = unsafe {
        PY_ARRAY_API.PyArray_CastToType(
            ob.py(),
            ob.as_ptr() as _,
            U::get_dtype(ob.py()).into_dtype_ptr(),
            0,
        )
    };
    if !ptr.is_null() {
        Ok(unsafe { Bound::from_owned_ptr(ob.py(), ptr).cast_into_unchecked() })
    } else {
        Err(PyErr::fetch(ob.py()))
    }
}

impl<'p> FromPyObject<'_, 'p> for Arg<'p, f64> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'p, PyAny>) -> PyResult<Self> {
        let ob = &*ob;
        if let Ok(value) = ob.extract::<f64>() {
            return Ok(Arg::Scalar(value));
        };

        if ob.cast::<PyList>().is_ok()
            || ob.cast::<PyTuple>().is_ok()
            || ob.cast::<PyIterator>().is_ok()
            || ob.cast::<PySequence>().is_ok()
        {
            let arr = pyiter_to_arrayd(ob.try_iter()?)?;
            return Ok(Arg::Array(CowArray::from(arr)));
        }

        if is_numpy_available() {
            if let Ok(a) = ob.cast::<numpy::PyArrayDyn<f64>>() {
                return Ok(Arg::NumpyArray(a.clone()));
            }

            if unsafe { npyffi::PyArray_Check(ob.py(), ob.as_ptr()) } == 1 {
                let a = pyarray_cast::<f64>(ob)?;
                return Ok(Arg::NumpyArray(a));
            }
        }

        Err(PyTypeError::new_err("must be float scalar or array-like"))
    }
}

impl<'p> FromPyObject<'_, 'p> for Arg<'p, bool> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'p, PyAny>) -> PyResult<Self> {
        let ob = &*ob;
        if let Ok(value) = ob.extract::<bool>() {
            return Ok(Arg::Scalar(value));
        };

        if ob.cast::<PyList>().is_ok()
            || ob.cast::<PyTuple>().is_ok()
            || ob.cast::<PyIterator>().is_ok()
            || ob.cast::<PySequence>().is_ok()
        {
            let arr = pyiter_to_arrayd(ob.try_iter()?)?;
            return Ok(Arg::Array(CowArray::from(arr)));
        }

        if is_numpy_available() {
            if let Ok(a) = ob.cast::<numpy::PyArrayDyn<bool>>() {
                return Ok(Arg::NumpyArray(a.clone()));
            }
        }

        Err(PyTypeError::new_err("must be bool scalar or array-like"))
    }
}

impl<'py> IntoPyObject<'py> for Arg<'py, f64> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Arg::Scalar(s) => Ok(float_or_none(s).into_pyobject(py)?),
            Arg::Array(a) => match arrayd_to_pylist(py, a.view()) {
                Ok(py_list) => Ok(py_list.into_any()),
                Err(err) => Err(err),
            },
            Arg::NumpyArray(a) => Ok(a.into_any().into_pyobject(py)?),
        }
    }
}

impl<'p, T> From<Arg<'p, T>> for CowArray<'p, T, IxDyn>
where
    T: Clone + numpy::Element,
{
    fn from(arg: Arg<'p, T>) -> Self {
        match arg {
            Arg::Scalar(value) => CowArray::from(ndarray::arr1(&[value]).into_dyn()),
            Arg::Array(a) => a,
            Arg::NumpyArray(a) => {
                // let b = unsafe { a.as_array() };
                // CowArray::from(b)
                CowArray::from(a.to_owned_array())
            } // Arg::NumpyArray(a) => CowArray::from(unsafe { a.as_array() }),
        }
    }
}

impl<T> From<ArrayD<T>> for Arg<'_, T> {
    fn from(arr: ArrayD<T>) -> Self {
        Arg::Array(CowArray::from(arr))
    }
}

impl<'p, T> From<Bound<'p, PyArrayDyn<T>>> for Arg<'p, T> {
    fn from(arr: Bound<'p, PyArrayDyn<T>>) -> Self {
        Arg::NumpyArray(arr)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::ffi::c_str;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[&[3_usize, 2], &[2, 1]]), None);
        assert_eq!(broadcast_shapes(&[&[1_usize, 2], &[3, 1], &[3, 2]]), Some(vec![3_usize, 2]));
        assert_eq!(
            broadcast_shapes(&[&[6_usize, 7], &[5, 6, 1], &[7], &[5, 1, 7]]),
            Some(vec![5, 6, 7])
        );
    }

    #[rstest]
    fn test_flatten_pyiter() {
        Python::attach(|py| {
            let ob = py.eval(c_str!("(range(i, i + 3) for i in range(3))"), None, None).unwrap();
            let array = pyiter_to_arrayd::<i64>(ob.try_iter().unwrap()).unwrap();
            let expected = ndarray::array![[0, 1, 2], [1, 2, 3], [2, 3, 4]].into_dyn();
            assert!(array == expected)
        });
    }

    #[rstest]
    #[case("[]", vec![0])]
    #[case("[[]]", vec![1, 0])]
    #[case("[[], []]", vec![2, 0])]
    #[case("[[[], []], [[], []]]", vec![2, 2, 0])]
    #[case("[[1, 2, 3], [4, 5, 6]]", vec![2, 3])]
    fn test_rectangular_shapes(#[case] input: &str, #[case] shape: Vec<usize>) {
        Python::attach(|py| {
            let input = std::ffi::CString::new(input).unwrap();
            let value = py.eval(&input, None, None).unwrap();
            let array = pyiter_to_arrayd::<i64>(value.try_iter().unwrap()).unwrap();
            assert_eq!(array.shape(), shape);
        });
    }

    #[rstest]
    #[case("[[], [[], []]]")]
    #[case("[1, []]")]
    #[case("[[1, 2], [3]]")]
    fn test_rejects_ragged_shapes(#[case] input: &str) {
        Python::attach(|py| {
            let input = std::ffi::CString::new(input).unwrap();
            let value = py.eval(&input, None, None).unwrap();
            let error = pyiter_to_arrayd::<i64>(value.try_iter().unwrap()).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    #[case("['1']")]
    #[case("'invalid'")]
    fn test_rejects_strings(#[case] input: &str) {
        Python::attach(|py| {
            let input = std::ffi::CString::new(input).unwrap();
            let value = py.eval(&input, None, None).unwrap();
            let error = pyiter_to_arrayd::<f64>(value.try_iter().unwrap()).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[rstest]
    fn test_rejects_cycles_but_accepts_shared_lists() {
        Python::attach(|py| {
            let child = PyList::new(py, [1.0, 2.0]).unwrap();
            let shared = PyList::new(py, [&child, &child]).unwrap();
            let array = pyiter_to_arrayd::<f64>(shared.try_iter().unwrap()).unwrap();
            assert_eq!(array, ndarray::array![[1.0, 2.0], [1.0, 2.0]].into_dyn());

            let cyclic = PyList::empty(py);
            cyclic.append(&cyclic).unwrap();
            let error = pyiter_to_arrayd::<f64>(cyclic.try_iter().unwrap()).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    fn test_nesting_limit() {
        Python::attach(|py| {
            let mut value = 0.1.into_py_any(py).unwrap().into_bound(py);
            for _ in 0..MAX_ARRAY_DIMS {
                value = PyList::new(py, [value]).unwrap().into_any();
            }
            let array = pyiter_to_arrayd::<f64>(value.try_iter().unwrap()).unwrap();
            assert_eq!(array.ndim(), MAX_ARRAY_DIMS);
            let deeper = PyList::new(py, [value]).unwrap();
            let error = pyiter_to_arrayd::<f64>(deeper.try_iter().unwrap()).unwrap_err();
            assert!(error.is_instance_of::<PyRecursionError>(py));
        });
    }
}

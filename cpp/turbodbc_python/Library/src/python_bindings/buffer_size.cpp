#include <turbodbc/buffer_size.h>

#include <pybind11/pybind11.h>

//#include <boost/python/class.hpp>
#include <boost/variant.hpp>

namespace turbodbc { namespace bindings {

//struct buffer_size_to_object : boost::static_visitor<PyObject *> {
//    static PyObject * convert(buffer_size const& b) {
//        return apply_visitor(buffer_size_to_object(), b);
//    }
//
//    template<typename T>
//    PyObject * operator()(T const& t) const {
//        return boost::python::incref(boost::python::object(t).ptr());
//    }
//};
//
//struct buffer_size_from_object{
//    static bool is_convertible (PyObject * object)
//    {
//        return boost::python::extract<turbodbc::rows>(object).check()
//            or boost::python::extract<turbodbc::megabytes>(object).check();
//    }
//
//    static buffer_size convert(PyObject * object)
//    {
//        boost::python::object python_value{boost::python::handle<>{boost::python::borrowed(object)}};
//
//        {
//            boost::python::extract<turbodbc::rows> extractor(object);
//            if (extractor.check()) {
//                return turbodbc::rows(extractor());
//            }
//        }
//        {
//            boost::python::extract<turbodbc::megabytes> extractor(object);
//            return turbodbc::megabytes(extractor());
//        }
//    }
//};
//
//template<typename Converter> struct boost_python_converter
//{
//	using target = decltype(Converter::convert(std::declval<PyObject*>()));
//
//	static void * is_convertible(PyObject* object)
//	{
//		if(Converter::is_convertible(object)){
//			return object;
//		} else {
//			return nullptr;
//		}
//	}
//
//	static void convert(PyObject* object, boost::python::converter::rvalue_from_python_stage1_data* data)
//	{
//		void* storage = (reinterpret_cast<boost::python::converter::rvalue_from_python_storage<target>*>(data))->storage.bytes;
//		new (storage) target(Converter::convert(object));
//		data->convertible = storage;
//	}
//};

//namespace pybind11 { namespace detail {
//
//template <> struct type_caster<buffer_size> {
//public:
//	/**
//	 * This macro establishes the name 'inty' in
//	 * function signatures and declares a local variable
//	 * 'value' of type inty
//	 */
//	PYBIND11_TYPE_CASTER(buffer_size, _("Megabytes"));
//
//	/**
//	 * Conversion part 1 (Python->C++): convert a PyObject into a inty
//	 * instance or return false upon failure. The second argument
//	 * indicates whether implicit conversions should be applied.
//	 */
//	bool load(handle src, bool) {
//		/* Extract PyObject from handle */
//		PyObject *source = src.ptr();
//		/* Try converting into a Python integer value */
//		PyObject *tmp = PyNumber_Long(source);
//		if (!tmp)
//			return false;
//		/* Now try to convert into a C++ int */
//		value.long_value = PyLong_AsLong(tmp);
//		Py_DECREF(tmp);
//		/* Ensure return code was OK (to avoid out-of-range errors etc) */
//		return !(value.long_value == -1 && !PyErr_Occurred());
//	}
//
//	/**
//	 * Conversion part 2 (C++ -> Python): convert an inty instance into
//	 * a Python object. The second and third arguments are used to
//	 * indicate the return value policy and parent object (for
//	 * ``return_value_policy::reference_internal``) and are generally
//	 * ignored by implicit casters.
//	 */
//	static handle cast(inty src, return_value_policy /* policy */, handle /* parent */) {
//		return PyLong_FromLong(src.long_value);
//	}
//};
//
//
//}}

void for_buffer_size(pybind11::module & module)
{
    pybind11::class_<turbodbc::rows>(module, "Rows")
		.def(pybind11::init<std::size_t>())
		.def_readwrite("rows", &turbodbc::rows::value)
    ;

    pybind11::class_<turbodbc::megabytes>(module, "Megabytes")
		.def(pybind11::init<std::size_t>())
		.def_readwrite("megabytes", &turbodbc::megabytes::value)
    ;

}

} }
#include <turbodbc/connection.h>

#include <pybind11/pybind11.h>

#include <boost/variant.hpp>

// The following conversion logic is adapted from https://github.com/pybind/pybind11/issues/576
// it is placed in this file because it is the only place where a function requiring
// conversion to/from turbodbc::buffer_size is used. There is no global converter registry,
// so the only other way to place the conversion logic is in a header file.
namespace pybind11 { namespace detail {

struct variant_caster_visitor : boost::static_visitor<handle> {
	variant_caster_visitor(return_value_policy policy, handle parent) :
		policy(policy),
		parent(parent)
	{}

	return_value_policy policy;
	handle parent;

	template<class T>
	handle operator()(T const& src) const {
		return make_caster<T>::cast(src, policy, parent);
	}
};

template <>
struct type_caster<turbodbc::buffer_size> {
	using Type = turbodbc::buffer_size;

	PYBIND11_TYPE_CASTER(Type, _("BufferSize"));

	template<class T>
	bool try_load(handle py_value, bool convert) {
		auto caster = make_caster<T>();
		if (caster.load(py_value, convert)) {
			value = cast_op<T>(caster);
			return true;
		}
		return false;
	}

	bool load(handle py_value, bool convert) {
		return (try_load<turbodbc::megabytes>(py_value, convert)) or
		       (try_load<turbodbc::rows>(py_value, convert));
	}

	static handle cast(Type const & cpp_value, return_value_policy policy, handle parent) {
		return boost::apply_visitor(variant_caster_visitor(policy, parent), cpp_value);
	}
};

} }


namespace turbodbc { namespace bindings {

void for_connection(pybind11::module & module)
{
	pybind11::class_<turbodbc::connection>(module, "Connection")
			.def("commit", &turbodbc::connection::commit)
			.def("rollback", &turbodbc::connection::rollback)
			.def("cursor", &turbodbc::connection::make_cursor)
			.def("get_buffer_size", &turbodbc::connection::get_buffer_size)
			.def("set_buffer_size", &turbodbc::connection::set_buffer_size)
			.def_readwrite("parameter_sets_to_buffer", &turbodbc::connection::parameter_sets_to_buffer)
			.def_readwrite("use_async_io", &turbodbc::connection::use_async_io)
			;
}

} }

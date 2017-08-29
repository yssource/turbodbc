#pragma once

#include <turbodbc/parameter_sets/bound_parameter_set.h>

#undef BOOL
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <pybind11/pybind11.h>

namespace turbodbc_arrow {

void set_arrow_parameters(turbodbc::bound_parameter_set & parameters, pybind11::object const & pyarrow_table);

}

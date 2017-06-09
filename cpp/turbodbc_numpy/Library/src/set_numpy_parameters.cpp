#include <turbodbc_numpy/set_numpy_parameters.h>

#include <turbodbc/errors.h>

#include <iostream>

namespace turbodbc_numpy {


void set_numpy_parameters(turbodbc::bound_parameter_set & parameters, std::vector<pybind11::array> const & columns)
{
    pybind11::dtype const np_int64("int64");
    pybind11::dtype const np_int16("int16");

    auto const & column = columns.front();
    auto const dtype = column.dtype();

    if (dtype == np_int64) {
        return;
    }

    throw turbodbc::interface_error("Encountered unsupported NumPy dtype '" +
                                    static_cast<std::string>(pybind11::str(dtype)) + "'");
}

}
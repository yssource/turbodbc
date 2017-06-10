#include <turbodbc_numpy/set_numpy_parameters.h>

#include <turbodbc/errors.h>
#include <turbodbc/make_description.h>
#include <turbodbc/type_code.h>

#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif
#include <sql.h>

namespace turbodbc_numpy {


void set_numpy_parameters(turbodbc::bound_parameter_set & parameters, std::vector<pybind11::array> const & columns)
{
    pybind11::dtype const np_int64("int64");

    auto const & column = columns.front();
    auto const dtype = column.dtype();

    if (dtype == np_int64) {
        auto unchecked = column.unchecked<std::int64_t, 1>();
        auto data_ptr = unchecked.data(0);
        parameters.rebind(0, turbodbc::make_description(turbodbc::type_code::integer, 0));
        auto & buffer = parameters.get_parameters()[0]->get_buffer();
        std::memcpy(buffer.data_pointer(), data_ptr, unchecked.size() * sizeof(std::int64_t));
        std::fill_n(buffer.indicator_pointer(), unchecked.size(), static_cast<intptr_t>(sizeof(std::int64_t)));
        parameters.execute_batch(unchecked.size());
    } else {
        throw turbodbc::interface_error("Encountered unsupported NumPy dtype '" +
                                        static_cast<std::string>(pybind11::str(dtype)) + "'");
    }
}

}
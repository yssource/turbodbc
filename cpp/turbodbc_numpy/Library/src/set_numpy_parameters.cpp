#include <turbodbc_numpy/set_numpy_parameters.h>

#include <turbodbc_numpy/ndarrayobject.h>

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

namespace {

    template<typename Value>
    void set_batch(turbodbc::parameter & parameter, pybind11::array const & column, pybind11::array_t<bool> const & mask, std::size_t start, std::size_t elements)
    {
        auto unchecked = column.unchecked<Value, 1>();
        auto data_ptr = unchecked.data(0);
        auto & buffer = parameter.get_buffer();
        std::memcpy(buffer.data_pointer(), data_ptr + start, elements * sizeof(Value));
        if (mask.size() != 1) {
            auto const indicator = buffer.indicator_pointer();
            auto const mask_start = mask.unchecked<1>().data(start);
            for (std::size_t i = 0; i != elements; ++i) {
                *(indicator + i) = (*(mask_start + i) == NPY_TRUE) ? SQL_NULL_DATA : 7;
            }
        } else {
            std::fill_n(buffer.indicator_pointer(), elements, static_cast<intptr_t>(sizeof(Value)));
        }
    }

}

void set_numpy_parameters(turbodbc::bound_parameter_set & parameters, std::vector<std::tuple<pybind11::array, pybind11::array_t<bool>>> const & columns)
{
    if (parameters.number_of_parameters() != columns.size()) {
        throw turbodbc::interface_error("Number of passed columns is not equal to the number of parameters");
    }

    if (columns.size() == 0) {
        return;
    }

    pybind11::dtype const np_int64("int64");
    auto const total_sets = std::get<0>(columns.front()).size();

    for (std::size_t i = 0; i != columns.size(); ++i) {
        auto const dtype = std::get<0>(columns[i]).dtype();
        if (dtype == np_int64) {
            parameters.rebind(i, turbodbc::make_description(turbodbc::type_code::integer, 0));
        } else {
            throw turbodbc::interface_error("Encountered unsupported NumPy dtype '" +
                                            static_cast<std::string>(pybind11::str(dtype)) + "'");
        }
    }

    for (std::size_t start = 0; start < total_sets; start += parameters.buffered_sets()) {
        auto const in_this_batch = std::min(parameters.buffered_sets(), total_sets - start);
        for (std::size_t i = 0; i != columns.size(); ++i) {
            auto const & data = std::get<0>(columns[i]);
            auto const & mask = std::get<1>(columns[i]);
            set_batch<std::int64_t>(*parameters.get_parameters()[i], data, mask, start, in_this_batch);
        }
        parameters.execute_batch(in_this_batch);
    }
}

}
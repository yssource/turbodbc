#include <turbodbc_arrow/set_arrow_parameters.h>

#include <turbodbc/errors.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include <sql.h>

using arrow::ChunkedArray;
using arrow::Int64Type;
using arrow::NumericArray;
using arrow::Table;

namespace turbodbc_arrow {

namespace {

    struct parameter_converter {
        parameter_converter(std::shared_ptr<ChunkedArray> const & data,
                            turbodbc::bound_parameter_set & parameters,
                            std::size_t parameter_index) :
            data(data),
            parameters(parameters),
            parameter_index(parameter_index)
        {}

        cpp_odbc::multi_value_buffer & get_buffer() {
            return parameters.get_parameters()[parameter_index]->get_buffer();
        }
        virtual void set_batch(int64_t start, int64_t elements) = 0;

        virtual ~parameter_converter() = default;

        std::shared_ptr<ChunkedArray> const & data;
        turbodbc::bound_parameter_set & parameters;
        std::size_t const parameter_index;
    };

    struct int64_converter : public parameter_converter {
      using parameter_converter::parameter_converter;

      void set_batch(int64_t start, int64_t elements) override {
        auto & buffer = get_buffer();

        if (data->num_chunks() != 1) {
          throw turbodbc::interface_error("Chunked int64 columns are not yet supported");
        }

        auto const& typed_array = static_cast<const NumericArray<Int64Type>&>(*data->chunk(0));
        int64_t const* data_ptr = typed_array.raw_values();
        memcpy(buffer.data_pointer(), data_ptr + start, elements * sizeof(int64_t));

        if (typed_array.null_count() == 0) {
          std::fill_n(buffer.indicator_pointer(), elements, sizeof(int64_t));
        } else if (typed_array.null_count() == typed_array.length()) {
          std::fill_n(buffer.indicator_pointer(), elements, SQL_NULL_DATA);
        } else {
          auto const indicator = buffer.indicator_pointer();
          for (int64_t i = 0; i != elements; ++i) {
            indicator[i] = typed_array.IsNull(start + i) ? SQL_NULL_DATA : sizeof(int64_t);
          }
        }
      };
    };

    std::vector<std::unique_ptr<parameter_converter>> make_converters(
        Table const & table,
        turbodbc::bound_parameter_set & parameters)
    {
        std::vector<std::unique_ptr<parameter_converter>> converters;

        for (int64_t i = 0; i < table.num_columns(); ++i) {
            std::shared_ptr<ChunkedArray> const & data = table.column(i)->data();
            arrow::Type::type dtype = data->type()->id();

            switch (dtype) {
              case arrow::Type::INT64:
                converters.emplace_back(new int64_converter(data, parameters, i));
                break;
              case arrow::Type::DOUBLE:
              case arrow::Type::STRING:
              // TODO: timestamp, date, bool
              default:
                std::ostringstream message;
                message << "Unsupported Arrow type for column " << (i + 1) << " of ";
                message << table.num_columns() << " (" << data->type()->ToString() << ")";
                throw turbodbc::interface_error(message.str());
            }
        }

        return converters;
    }
}

void set_arrow_parameters(turbodbc::bound_parameter_set & parameters, pybind11::object const & pyarrow_table) {
  arrow::py::import_pyarrow();
  if (arrow::py::is_table(pyarrow_table.ptr())) {
    std::shared_ptr<Table> table;
    // TODO: Check status
    arrow::py::unwrap_table(pyarrow_table.ptr(), &table);

    if (static_cast<int32_t>(parameters.number_of_parameters()) != table->num_columns()) {
        std::stringstream ss;
        ss << "Number of passed columns (" << table->num_columns();
        ss << ") is not equal to the number of parameters (";
        ss << parameters.number_of_parameters() << ")";
        throw turbodbc::interface_error(ss.str());
    }

    if (table->num_columns() == 0) {
        return;
    }

    auto converters = make_converters(*table, parameters);
    std::size_t const total_sets = static_cast<std::size_t>(table->num_rows());

    for (std::size_t start = 0; start < total_sets; start += parameters.buffered_sets()) {
        auto const in_this_batch = std::min(parameters.buffered_sets(), total_sets - start);
        for (int64_t i = 0; i < table->num_columns(); ++i) {
            // converters[i]->set_batch(start, in_this_batch);
        }
        parameters.execute_batch(in_this_batch);
    }

  }
}

}

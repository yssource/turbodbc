#include <turbodbc_arrow/set_arrow_parameters.h>

#include <turbodbc/errors.h>
#include <turbodbc/make_description.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include <sql.h>

using arrow::BooleanArray;
using arrow::BinaryArray;
using arrow::ChunkedArray;
using arrow::Int64Type;
using arrow::NumericArray;
using arrow::StringArray;
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
        
        template <size_t element_size>
        void set_indicator(cpp_odbc::multi_value_buffer & buffer, int64_t start, int64_t elements) {
            if (data->num_chunks() != 1) {
              throw turbodbc::interface_error("Chunked columns are not yet supported");
            }

            arrow::Array const& chunk = *data->chunk(0);
            if (chunk.null_count() == 0) {
              std::fill_n(buffer.indicator_pointer(), elements, element_size);
            } else if (chunk.null_count() == chunk.length()) {
              std::fill_n(buffer.indicator_pointer(), elements, SQL_NULL_DATA);
            } else {
              auto const indicator = buffer.indicator_pointer();
              for (int64_t i = 0; i != elements; ++i) {
                indicator[i] = chunk.IsNull(start + i) ? SQL_NULL_DATA : element_size;
              }
            }
        }

        virtual void set_batch(int64_t start, int64_t elements) = 0;

        virtual ~parameter_converter() = default;

        std::shared_ptr<ChunkedArray> const & data;
        turbodbc::bound_parameter_set & parameters;
        std::size_t const parameter_index;
    };

    struct string_converter : public parameter_converter {
        string_converter(std::shared_ptr<ChunkedArray> const & data,
                         turbodbc::bound_parameter_set & parameters,
                         std::size_t parameter_index) :
            parameter_converter(data, parameters, parameter_index),
            type(parameters.get_initial_parameter_types()[parameter_index])
        {}
        
        template <typename String>
        void set_batch_of_type(std::size_t start, std::size_t elements)
        {
            if (data->num_chunks() != 1) {
              throw turbodbc::interface_error("Chunked int64 columns are not yet supported");
            }
        
            auto const& typed_array = static_cast<const BinaryArray&>(*data->chunk(0));

            int32_t maximum_length = 0;
            for (int64_t i = 0; i != elements; ++i) {
                if (!typed_array.IsNull(start + i)) {
                    maximum_length = std::max(maximum_length, typed_array.value_length(start + i));
                }
            }

            // Propagate the maximum string length to the parameters.
            // These then adjust the size of the underlying buffer.
            parameters.rebind(parameter_index, turbodbc::make_description(type, maximum_length));
            auto & buffer = get_buffer();
            auto const character_size = sizeof(typename String::value_type);

            for (int64_t i = 0; i != elements; ++i) {
                auto element = buffer[i];
                if (typed_array.IsNull(start + i)) {
                    element.indicator = SQL_NULL_DATA;
                } else {
                    int32_t out_length;
                    const uint8_t *value = typed_array.GetValue(start + i, &out_length);
                    std::memcpy(element.data_pointer, value, out_length);
                    element.indicator = character_size * typed_array.value_length(start + i);
                }
            }
        }

        void set_batch(int64_t start, int64_t elements) final
        {
            if (type == turbodbc::type_code::unicode) {
                throw turbodbc::interface_error("UTF-16 Strings are not supported yet");
                // set_batch_of_type<std::u16string>(start, elements);
            } else {
                set_batch_of_type<std::string>(start, elements);
            }
        }

    private:
        turbodbc::type_code type;
    };

    struct int64_converter : public parameter_converter {
      using parameter_converter::parameter_converter;

      void set_batch(int64_t start, int64_t elements) final {
        // parameters.rebind(parameter_index, turbodbc::make_description(turbodbc::type_code::integer, 0));
        auto & buffer = get_buffer();

        if (data->num_chunks() != 1) {
          throw turbodbc::interface_error("Chunked int64 columns are not yet supported");
        }

        auto const& typed_array = static_cast<const NumericArray<Int64Type>&>(*data->chunk(0));
        int64_t const* data_ptr = typed_array.raw_values();
        memcpy(buffer.data_pointer(), data_ptr + start, elements * sizeof(int64_t));

        set_indicator<sizeof(int64_t)>(buffer, start, elements);
      }
    };

    struct bool_converter : public parameter_converter {
        bool_converter(std::shared_ptr<ChunkedArray> const & data,
                         turbodbc::bound_parameter_set & parameters,
                         std::size_t parameter_index) :
            parameter_converter(data, parameters, parameter_index)
        {
          parameters.rebind(parameter_index, turbodbc::make_description(turbodbc::type_code::boolean, 0));
        }

      void set_batch(int64_t start, int64_t elements) final {
        if (data->num_chunks() != 1) {
          throw turbodbc::interface_error("Chunked int64 columns are not yet supported");
        }

        auto & buffer = get_buffer();
        auto const& typed_array = static_cast<const BooleanArray&>(*data->chunk(0));
        if (typed_array.null_count() < typed_array.length()) {
          for (int64_t i = 0; i != elements; ++i) {
            if (not typed_array.IsNull(start + i)) {
              buffer.data_pointer()[i] = static_cast<int8_t>(typed_array.Value(start + i));
            }
          }
        }
        
        set_indicator<sizeof(bool)>(buffer, start, elements);
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
              case arrow::Type::BINARY:
              case arrow::Type::STRING:
                converters.emplace_back(new string_converter(data, parameters, i));
                break;
              case arrow::Type::BOOL:
                converters.emplace_back(new bool_converter(data, parameters, i));
                break;
              // TODO: timestamp, date, bool
              case arrow::Type::DOUBLE:
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
            converters[i]->set_batch(start, in_this_batch);
        }
        parameters.execute_batch(in_this_batch);
    }

  }
}

}

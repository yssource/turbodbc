#include <turbodbc_arrow/arrow_result_set.h>

// Somewhere a macro defines BOOL as a constant. This is in conflict with array/type.h
#undef BOOL
#include <arrow/column.h>
#include <arrow/schema.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/types/primitive.h>
#include <arrow/types/string.h>
#include <arrow/util/memory-pool.h>

#include <boost/python/list.hpp>

#include <sql.h>

#include <vector>

/**
 * TODO
 *
 * RETURN_NOT_OK
 */

using arrow::default_memory_pool;
using arrow::DoubleBuilder;
using arrow::Int64Builder;
using arrow::BooleanBuilder;
using arrow::TimestampBuilder;
using arrow::StringBuilder;
using arrow::ArrayBuilder;
using arrow::TimeUnit;

namespace turbodbc_arrow {


namespace {

	std::unique_ptr<ArrayBuilder> make_array_builder(turbodbc::type_code type)
	{
		switch (type) {
			case turbodbc::type_code::floating_point:
				return std::unique_ptr<ArrayBuilder>(new DoubleBuilder(default_memory_pool(), std::make_shared<arrow::DoubleType>()));
			case turbodbc::type_code::integer:
				return std::unique_ptr<ArrayBuilder>(new Int64Builder(default_memory_pool(), std::make_shared<arrow::Int64Type>()));
			case turbodbc::type_code::boolean:
				return std::unique_ptr<ArrayBuilder>(new BooleanBuilder(default_memory_pool(), std::make_shared<arrow::BooleanType>()));
			case turbodbc::type_code::timestamp:
				return std::unique_ptr<TimestampBuilder>(new TimestampBuilder(default_memory_pool(), std::make_shared<arrow::TimestampType>(TimeUnit::MICRO)));
			case turbodbc::type_code::date:
        return nullptr; // TODO unsupported
			default:
				return std::unique_ptr<StringBuilder>(new StringBuilder(default_memory_pool(), std::make_shared<arrow::StringType>()));
		}
	}

	boost::python::list as_python_list()
	{
		boost::python::list result;
		/*for (auto & object : objects) {
			result.append(boost::python::make_tuple(object->get_data(), object->get_mask()));
		}*/
		return result;
	}

  // Copied from turbodbc_numpy/src/datetime_column.cpp
  boost::posix_time::ptime const timestamp_epoch({1970, 1, 1}, {0, 0, 0, 0});

	long timestamp_to_microseconds(char const * data_pointer)
	{
		auto & sql_ts = *reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(data_pointer);
		long const microseconds = sql_ts.fraction / 1000;
		boost::posix_time::ptime const ts({static_cast<unsigned short>(sql_ts.year), sql_ts.month, sql_ts.day},
		                                  {sql_ts.hour, sql_ts.minute, sql_ts.second, microseconds});
		return (ts - timestamp_epoch).total_microseconds();
	}

	boost::gregorian::date const date_epoch(1970, 1, 1);

	long date_to_days(char const * data_pointer)
	{
		auto & sql_date = *reinterpret_cast<SQL_DATE_STRUCT const *>(data_pointer);
		boost::gregorian::date const date(sql_date.year, sql_date.month, sql_date.day);
		return (date - date_epoch).days();
	}
}

arrow_result_set::arrow_result_set(turbodbc::result_sets::result_set & base) :
	base_result_(base)
{
}

std::shared_ptr<arrow::Schema> arrow_result_set::schema()
{
	auto const column_info = base_result_.get_column_info();
  auto const n_columns = column_info.size();
  std::vector<std::shared_ptr<arrow::Field>> fields;
	for (std::size_t i = 0; i != n_columns; ++i) {
    std::shared_ptr<arrow::DataType> type;
		switch (column_info[i].type) {
			case turbodbc::type_code::floating_point:
        type = arrow::float64();
        break;
			case turbodbc::type_code::integer:
        type = arrow::int64();
        break;
			case turbodbc::type_code::boolean:
        type = arrow::boolean();
        break;
			case turbodbc::type_code::timestamp:
        type = arrow::timestamp(TimeUnit::MICRO);
        break;
			case turbodbc::type_code::date:
        type = arrow::date();
        break;
			default:
        type = std::make_shared<arrow::StringType>();
		}
    fields.emplace_back(std::make_shared<arrow::Field>(column_info[i].name, type, column_info[i].supports_null_values));
	}
  return std::make_shared<arrow::Schema>(fields);
}

arrow::Status arrow_result_set::fetch_all_native(std::shared_ptr<arrow::Table>* out)
{
	std::size_t rows_in_batch = base_result_.fetch_next_batch();
	auto const column_info = base_result_.get_column_info();
  auto const n_columns = column_info.size();

  std::shared_ptr<arrow::Schema> arrow_schema = schema();

	std::vector<std::unique_ptr<ArrayBuilder>> columns;
	for (std::size_t i = 0; i != n_columns; ++i) {
		columns.push_back(make_array_builder(column_info[i].type));
	}

	do {
		auto const buffers = base_result_.get_buffers();

    std::vector<uint8_t> valid_bytes(rows_in_batch);
		for (size_t i = 0; i != n_columns; ++i) {
	    auto const indicator_pointer = buffers[i].get().indicator_pointer();
      for (size_t element = 0; element != rows_in_batch; ++element) {
        if (indicator_pointer[element] == SQL_NULL_DATA) {
          valid_bytes[element] = 0;
        } else {
          valid_bytes[element] = 1;
        }
      }
	  	switch (column_info[i].type) {
	  		case turbodbc::type_code::floating_point:
          static_cast<DoubleBuilder*>(columns[i].get())->Append(reinterpret_cast<const double*>(buffers[i].get().data_pointer()), rows_in_batch, valid_bytes.data());
          break;
	  		case turbodbc::type_code::integer:
          static_cast<Int64Builder*>(columns[i].get())->Append(reinterpret_cast<const int64_t*>(buffers[i].get().data_pointer()), rows_in_batch, valid_bytes.data());
          break;
	  		case turbodbc::type_code::boolean:
          static_cast<BooleanBuilder*>(columns[i].get())->Append(reinterpret_cast<const uint8_t*>(buffers[i].get().data_pointer()), rows_in_batch, valid_bytes.data());
          break;
	  		case turbodbc::type_code::timestamp: {
            auto builder = static_cast<TimestampBuilder*>(columns[i].get());
            auto buffer = buffers[i];
	          for (std::size_t j = 0; j < rows_in_batch; ++j) {
              auto element = buffer.get()[j];
	          	if (element.indicator == SQL_NULL_DATA) {
                builder->AppendNull();
	          	} else {
	          		builder->Append(timestamp_to_microseconds(element.data_pointer));
	          	}
	          }
            break;
          }
	  		case turbodbc::type_code::date:
            // TODO Special date handling
            break;
	  		default: {
          auto builder = static_cast<StringBuilder*>(columns[i].get());
          auto buffer = buffers[i];
         	for (std::size_t j = 0; j != rows_in_batch; ++j) {
         		auto const element = buffer.get()[j];
         		if (element.indicator == SQL_NULL_DATA) {
              builder->AppendNull();
         		} else {
              builder->Append(element.data_pointer, element.indicator);
         		}
         	}
          break;
         }
	  	}
		}
		rows_in_batch = base_result_.fetch_next_batch();
	} while (rows_in_batch != 0);

  std::vector<std::shared_ptr<arrow::Column>> arrow_columns;
  for (size_t i = 0; i != n_columns; ++i) {
    std::shared_ptr<arrow::Array> array;
    columns[i]->Finish(&array);
    arrow_columns.emplace_back(std::make_shared<arrow::Column>(
          arrow_schema->field(i), array));
  }

  *out = std::make_shared<arrow::Table>("", arrow_schema, arrow_columns);
  return arrow::Status::OK();
}

boost::python::object arrow_result_set::fetch_all()
{
	return as_python_list();
}


}

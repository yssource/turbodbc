#include <turbodbc_arrow/arrow_result_set.h>

#undef BOOL
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/schema.h>
#include <arrow/test-util.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <sql.h>


struct mock_result_set : public turbodbc::result_sets::result_set
{
	MOCK_METHOD0(do_fetch_next_batch, std::size_t());
	MOCK_CONST_METHOD0(do_get_column_info, std::vector<turbodbc::column_info>());
	MOCK_CONST_METHOD0(do_get_buffers, std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>>());
};


TEST(ArrowResultSetTest, SimpleSchemaConversion)
{
	mock_result_set rs;
	std::vector<turbodbc::column_info> expected = {{
    "int_column", turbodbc::type_code::integer, true}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  turbodbc_arrow::arrow_result_set ars(rs);
  auto schema = ars.schema();
  ASSERT_EQ(schema->num_fields(), 1);
  auto field = schema->field(0);
  ASSERT_EQ(field->name, "int_column");
  ASSERT_EQ(field->type, arrow::int64());
  ASSERT_EQ(field->nullable, true);
}

TEST(ArrowResultSetTest, AllTypesSchemaConversion)
{
	mock_result_set rs;
	std::vector<turbodbc::column_info> expected = {
      {"float_column", turbodbc::type_code::floating_point, true},
      {"boolean_column", turbodbc::type_code::boolean, true},
      {"timestamp_column", turbodbc::type_code::timestamp, true},
      {"date_column", turbodbc::type_code::date, true},
      {"string_column", turbodbc::type_code::string, true},
      {"int_column", turbodbc::type_code::integer, true},
      {"nonnull_float_column", turbodbc::type_code::floating_point, false},
      {"nonnull_boolean_column", turbodbc::type_code::boolean, false},
      {"nonnull_timestamp_column", turbodbc::type_code::timestamp, false},
      {"nonnull_date_column", turbodbc::type_code::date, false},
      {"nonnull_string_column", turbodbc::type_code::string, false},
      {"nonnull_int_column", turbodbc::type_code::integer, false}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  std::vector<std::shared_ptr<arrow::Field>> expected_fields = {
    std::make_shared<arrow::Field>("float_column", arrow::float64()),
    std::make_shared<arrow::Field>("boolean_column", arrow::boolean()),
    std::make_shared<arrow::Field>("timestamp_column", arrow::timestamp(arrow::TimeUnit::MICRO)),
    std::make_shared<arrow::Field>("date_column", arrow::date()),
    std::make_shared<arrow::Field>("string_column", std::make_shared<arrow::StringType>()),
    std::make_shared<arrow::Field>("int_column", arrow::int64()),
    std::make_shared<arrow::Field>("nonnull_float_column", arrow::float64(), false),
    std::make_shared<arrow::Field>("nonnull_boolean_column", arrow::boolean(), false),
    std::make_shared<arrow::Field>("nonnull_timestamp_column", arrow::timestamp(arrow::TimeUnit::MICRO), false),
    std::make_shared<arrow::Field>("nonnull_date_column", arrow::date(), false),
    std::make_shared<arrow::Field>("nonnull_string_column", std::make_shared<arrow::StringType>(), false),
    std::make_shared<arrow::Field>("nonnull_int_column", arrow::int64(), false)
  };

  turbodbc_arrow::arrow_result_set ars(rs);
  auto schema = ars.schema();

  ASSERT_EQ(schema->num_fields(), 12);
  for (int i = 0; i < schema->num_fields(); i++) {
    EXPECT_TRUE(schema->field(i)->Equals(expected_fields[i]));
  }
}

TEST(ArrowResultSetTest, SingleBatchSingleColumnResultSetConversion)
{
	mock_result_set rs;
  const int64_t OUTPUT_SIZE = 100;

  // Expected output: a Table with a single column of int64s
  arrow::Int64Builder builder(arrow::default_memory_pool(), arrow::int64());
  for (int64_t i = 0; i < OUTPUT_SIZE; i++) {
    ASSERT_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> array;
  ASSERT_OK(builder.Finish(&array));
  std::shared_ptr<arrow::Int64Array> typed_array = std::static_pointer_cast<arrow::Int64Array>(array);
  std::vector<std::shared_ptr<arrow::Field>> fields({std::make_shared<arrow::Field>("int_column", arrow::int64(), true)});
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);
  std::vector<std::shared_ptr<arrow::Column>> columns ({std::make_shared<arrow::Column>(fields[0], array)});
  std::shared_ptr<arrow::Table> expected_table = std::make_shared<arrow::Table>("", schema, columns);

  // Mock schema
	std::vector<turbodbc::column_info> expected = {{
    "int_column", turbodbc::type_code::integer, true}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  // Mock output columns
  // * Single batch of 100 ints
	cpp_odbc::multi_value_buffer buffer(sizeof(int64_t), OUTPUT_SIZE);
  memcpy(buffer.data_pointer(), typed_array->data()->data(), sizeof(int64_t) * OUTPUT_SIZE);
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers = {buffer};
	EXPECT_CALL(rs, do_get_buffers()).WillOnce(testing::Return(expected_buffers));
  EXPECT_CALL(rs, do_fetch_next_batch()).WillOnce(testing::Return(OUTPUT_SIZE)).WillOnce(testing::Return(0));

  turbodbc_arrow::arrow_result_set ars(rs);
  std::shared_ptr<arrow::Table> table;
  ASSERT_OK(ars.fetch_all_native(&table));
  // TODO(ARROW-415): ASSERT_TRUE(expected_table->Equals(table));
  ASSERT_EQ(expected_table->name(), table->name());
  ASSERT_TRUE(expected_table->schema()->Equals(table->schema()));
  ASSERT_EQ(expected_table->num_columns(), table->num_columns());
  ASSERT_EQ(expected_table->num_rows(), table->num_rows());
  for (int i = 0; i < expected_table->num_columns(); i++) {
    // TODO(ARROW-416): ASSERT_TRUE(expected_table->column(i)->Equals(table->column(i)));
    ASSERT_EQ(expected_table->column(i)->length(), table->column(i)->length());
    ASSERT_EQ(expected_table->column(i)->null_count(), table->column(i)->null_count());
    ASSERT_TRUE(expected_table->column(i)->field()->Equals(table->column(i)->field()));
    std::shared_ptr<arrow::ChunkedArray> expected_c_array = expected_table->column(i)->data();
    std::shared_ptr<arrow::ChunkedArray> c_array = table->column(i)->data();
    // TODO(ARROW-417): ASSERT_TRUE(expected_c_array->Equals(c_array));
    // .. until then we assume a single chunk:
    ASSERT_EQ(c_array->num_chunks(), 1);
    ASSERT_TRUE(expected_c_array->chunk(0)->Equals(c_array->chunk(0)));
  }
}

TEST(ArrowResultSetTest, MultipleBatchSingleColumnResultSetConversion)
{
	mock_result_set rs;
  const int64_t OUTPUT_SIZE = 100;

  // Expected output: a Table with a single column of int64s
  arrow::Int64Builder builder(arrow::default_memory_pool(), arrow::int64());
  for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
    ASSERT_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> array;
  ASSERT_OK(builder.Finish(&array));
  std::shared_ptr<arrow::Int64Array> typed_array = std::static_pointer_cast<arrow::Int64Array>(array);
  std::vector<std::shared_ptr<arrow::Field>> fields({std::make_shared<arrow::Field>("int_column", arrow::int64(), true)});
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);
  std::vector<std::shared_ptr<arrow::Column>> columns ({std::make_shared<arrow::Column>(fields[0], array)});
  std::shared_ptr<arrow::Table> expected_table = std::make_shared<arrow::Table>("", schema, columns);

  // Mock schema
	std::vector<turbodbc::column_info> expected = {{
    "int_column", turbodbc::type_code::integer, true}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  // Mock output columns
  // * Two batches of 100 ints
	cpp_odbc::multi_value_buffer buffer_1(sizeof(int64_t), OUTPUT_SIZE);
  memcpy(buffer_1.data_pointer(), typed_array->data()->data(), sizeof(int64_t) * OUTPUT_SIZE);
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers_1 = {buffer_1};
	cpp_odbc::multi_value_buffer buffer_2(sizeof(int64_t), OUTPUT_SIZE);
  memcpy(buffer_2.data_pointer(), typed_array->data()->data() + sizeof(int64_t) * OUTPUT_SIZE, sizeof(int64_t) * OUTPUT_SIZE);
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers_2 = {buffer_2};
	EXPECT_CALL(rs, do_get_buffers()).WillOnce(testing::Return(expected_buffers_1)).WillOnce(testing::Return(expected_buffers_2));
  EXPECT_CALL(rs, do_fetch_next_batch()).WillOnce(testing::Return(OUTPUT_SIZE)).WillOnce(testing::Return(OUTPUT_SIZE)).WillOnce(testing::Return(0));

  turbodbc_arrow::arrow_result_set ars(rs);
  std::shared_ptr<arrow::Table> table;
  ASSERT_OK(ars.fetch_all_native(&table));
  // TODO(ARROW-415): ASSERT_TRUE(expected_table->Equals(table));
  ASSERT_EQ(expected_table->name(), table->name());
  ASSERT_TRUE(expected_table->schema()->Equals(table->schema()));
  ASSERT_EQ(expected_table->num_columns(), table->num_columns());
  ASSERT_EQ(expected_table->num_rows(), table->num_rows());
  for (int i = 0; i < expected_table->num_columns(); i++) {
    // TODO(ARROW-416): ASSERT_TRUE(expected_table->column(i)->Equals(table->column(i)));
    ASSERT_EQ(expected_table->column(i)->length(), table->column(i)->length());
    ASSERT_EQ(expected_table->column(i)->null_count(), table->column(i)->null_count());
    ASSERT_TRUE(expected_table->column(i)->field()->Equals(table->column(i)->field()));
    std::shared_ptr<arrow::ChunkedArray> expected_c_array = expected_table->column(i)->data();
    std::shared_ptr<arrow::ChunkedArray> c_array = table->column(i)->data();
    // TODO(ARROW-417): ASSERT_TRUE(expected_c_array->Equals(c_array));
    // .. until then we assume a single chunk:
    ASSERT_EQ(c_array->num_chunks(), 1);
    ASSERT_TRUE(expected_c_array->chunk(0)->Equals(c_array->chunk(0)));
  }
}

TEST(ArrowResultSetTest, MultipleBatchMultipleColumnResultSetConversion)
{
	mock_result_set rs;
  const int64_t OUTPUT_SIZE = 100;

  // Expected output: a Table with a single column of int64s
  arrow::Int64Builder builder(arrow::default_memory_pool(), arrow::int64());
  for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
    ASSERT_OK(builder.Append(i));
  }
  std::vector<std::shared_ptr<arrow::Field>> fields = {
    std::make_shared<arrow::Field>("float_column", arrow::float64()),
    std::make_shared<arrow::Field>("boolean_column", arrow::boolean()),
    std::make_shared<arrow::Field>("timestamp_column", arrow::timestamp(arrow::TimeUnit::MICRO)),
    std::make_shared<arrow::Field>("string_column", std::make_shared<arrow::StringType>()),
    std::make_shared<arrow::Field>("int_column", arrow::int64()),
    std::make_shared<arrow::Field>("nonnull_float_column", arrow::float64(), false),
    std::make_shared<arrow::Field>("nonnull_boolean_column", arrow::boolean(), false),
    std::make_shared<arrow::Field>("nonnull_timestamp_column", arrow::timestamp(arrow::TimeUnit::MICRO), false),
    std::make_shared<arrow::Field>("nonnull_string_column", std::make_shared<arrow::StringType>(), false),
    std::make_shared<arrow::Field>("nonnull_int_column", arrow::int64(), false)
    // std::make_shared<arrow::Field>("date_column", arrow::date()),
    // std::make_shared<arrow::Field>("nonnull_date_column", arrow::date(), false),
  };
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);

  std::vector<std::shared_ptr<arrow::Column>> columns;
	cpp_odbc::multi_value_buffer buffer_0(sizeof(double), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_0_2(sizeof(double), OUTPUT_SIZE);
  {
    arrow::DoubleBuilder builder(arrow::default_memory_pool(), arrow::float64());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      ASSERT_OK(builder.Append(i));
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[0], array));
    auto typed_array = static_cast<arrow::DoubleArray*>(array.get());
    memcpy(buffer_0.data_pointer(), typed_array->data()->data(), sizeof(double) * OUTPUT_SIZE);
    memcpy(buffer_0_2.data_pointer(), typed_array->data()->data() + sizeof(double) * OUTPUT_SIZE, sizeof(double) * OUTPUT_SIZE);
  }
	cpp_odbc::multi_value_buffer buffer_1(sizeof(bool), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_1_2(sizeof(bool), OUTPUT_SIZE);
  {
    arrow::BooleanBuilder builder(arrow::default_memory_pool(), arrow::boolean());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      ASSERT_OK(builder.Append(i % 3 == 0));
      if (i < OUTPUT_SIZE) {
        *(buffer_1[i].data_pointer) = (i % 3 == 0);
      } else {
        *(buffer_1_2[i - OUTPUT_SIZE].data_pointer) = (i % 3 == 0);
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[1], array));
  }
	cpp_odbc::multi_value_buffer buffer_2(sizeof(SQL_TIMESTAMP_STRUCT), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_2_2(sizeof(SQL_TIMESTAMP_STRUCT), OUTPUT_SIZE);
  {
    arrow::TimestampBuilder builder(arrow::default_memory_pool(), arrow::timestamp(arrow::TimeUnit::MICRO));
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      ASSERT_OK(builder.Append(i));
      auto td = boost::posix_time::microseconds(i);
      auto ts = boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1), td);
      SQL_TIMESTAMP_STRUCT* sql_ts;
      if (i < OUTPUT_SIZE) {
        sql_ts = reinterpret_cast<SQL_TIMESTAMP_STRUCT*>(buffer_2.data_pointer()) + i;
      } else {
        sql_ts = reinterpret_cast<SQL_TIMESTAMP_STRUCT*>(buffer_2_2.data_pointer()) + (i - OUTPUT_SIZE);
      }
      sql_ts->year = ts.date().year();
      sql_ts->month = ts.date().month();
      sql_ts->day = ts.date().day();
      sql_ts->hour = ts.time_of_day().hours();
      sql_ts->minute = ts.time_of_day().minutes();
      sql_ts->second = ts.time_of_day().seconds();
      sql_ts->fraction = ts.time_of_day().fractional_seconds() * 1000;
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[2], array));
  }
  // Longest string: "200" -> 4 bytes
	cpp_odbc::multi_value_buffer buffer_3(4, OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_3_2(4, OUTPUT_SIZE);
  {
    arrow::StringBuilder builder(arrow::default_memory_pool(), std::make_shared<arrow::StringType>());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      std::string str = std::to_string(i);
      ASSERT_OK(builder.Append(str));
        if (i < OUTPUT_SIZE) {
          memcpy(buffer_3[i].data_pointer, str.c_str(), str.size() + 1);
          buffer_3[i].indicator = str.size();
        } else {
          memcpy(buffer_3_2[i - OUTPUT_SIZE].data_pointer, str.c_str(), str.size() + 1);
          buffer_3_2[i - OUTPUT_SIZE].indicator = str.size();
        }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[3], array));
  }
	cpp_odbc::multi_value_buffer buffer_4(sizeof(int64_t), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_4_2(sizeof(int64_t), OUTPUT_SIZE);
  {
    arrow::Int64Builder builder(arrow::default_memory_pool(), arrow::int64());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      ASSERT_OK(builder.Append(i));
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[4], array));
    auto typed_array = static_cast<arrow::Int64Array*>(array.get());
    memcpy(buffer_4.data_pointer(), typed_array->data()->data(), sizeof(int64_t) * OUTPUT_SIZE);
    memcpy(buffer_4_2.data_pointer(), typed_array->data()->data() + sizeof(int64_t) * OUTPUT_SIZE, sizeof(int64_t) * OUTPUT_SIZE);
  }
	cpp_odbc::multi_value_buffer buffer_5(sizeof(double), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_5_2(sizeof(double), OUTPUT_SIZE);
  {
    arrow::DoubleBuilder builder(arrow::default_memory_pool(), arrow::float64());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      if (i % 5 == 0) {
        ASSERT_OK(builder.AppendNull());
        if (i < OUTPUT_SIZE) {
          buffer_5.indicator_pointer()[i] = SQL_NULL_DATA;
        } else {
          buffer_5_2.indicator_pointer()[i - OUTPUT_SIZE] = SQL_NULL_DATA;
        }
      } else {
        ASSERT_OK(builder.Append(i));
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[5], array));
    auto typed_array = static_cast<arrow::DoubleArray*>(array.get());
    memcpy(buffer_5.data_pointer(), typed_array->data()->data(), sizeof(double) * OUTPUT_SIZE);
    memcpy(buffer_5_2.data_pointer(), typed_array->data()->data() + sizeof(double) * OUTPUT_SIZE, sizeof(double) * OUTPUT_SIZE);
  }
	cpp_odbc::multi_value_buffer buffer_6(sizeof(bool), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_6_2(sizeof(bool), OUTPUT_SIZE);
  {
    arrow::BooleanBuilder builder(arrow::default_memory_pool(), arrow::boolean());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      if (i % 5 == 0) {
        ASSERT_OK(builder.AppendNull());
        if (i < OUTPUT_SIZE) {
          buffer_6.indicator_pointer()[i] = SQL_NULL_DATA;
        } else {
          buffer_6_2.indicator_pointer()[i - OUTPUT_SIZE] = SQL_NULL_DATA;
        }
      } else {
        ASSERT_OK(builder.Append(i % 3 == 0));
        if (i < OUTPUT_SIZE) {
          *(buffer_6[i].data_pointer) = (i % 3 == 0);
        } else {
          *(buffer_6_2[i - OUTPUT_SIZE].data_pointer) = (i % 3 == 0);
        }
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[6], array));
  }
	cpp_odbc::multi_value_buffer buffer_7(sizeof(SQL_TIMESTAMP_STRUCT), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_7_2(sizeof(SQL_TIMESTAMP_STRUCT), OUTPUT_SIZE);
  {
    arrow::TimestampBuilder builder(arrow::default_memory_pool(), arrow::timestamp(arrow::TimeUnit::MICRO));
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      if (i % 5 == 0) {
        ASSERT_OK(builder.AppendNull());
        if (i < OUTPUT_SIZE) {
          buffer_7.indicator_pointer()[i] = SQL_NULL_DATA;
        } else {
          buffer_7_2.indicator_pointer()[i - OUTPUT_SIZE] = SQL_NULL_DATA;
        }
      } else {
        ASSERT_OK(builder.Append(i));
        auto td = boost::posix_time::microseconds(i);
        auto ts = boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1), td);
        SQL_TIMESTAMP_STRUCT* sql_ts;
        if (i < OUTPUT_SIZE) {
          sql_ts = reinterpret_cast<SQL_TIMESTAMP_STRUCT*>(buffer_7.data_pointer()) + i;
        } else {
          sql_ts = reinterpret_cast<SQL_TIMESTAMP_STRUCT*>(buffer_7_2.data_pointer()) + (i - OUTPUT_SIZE);
        }
        sql_ts->year = ts.date().year();
        sql_ts->month = ts.date().month();
        sql_ts->day = ts.date().day();
        sql_ts->hour = ts.time_of_day().hours();
        sql_ts->minute = ts.time_of_day().minutes();
        sql_ts->second = ts.time_of_day().seconds();
        sql_ts->fraction = ts.time_of_day().fractional_seconds() * 1000;
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[7], array));
  }
  // Longest string: "200" -> 4 bytes
	cpp_odbc::multi_value_buffer buffer_8(4, OUTPUT_SIZE);
  cpp_odbc::multi_value_buffer buffer_8_2(4, OUTPUT_SIZE);
  {
    arrow::StringBuilder builder(arrow::default_memory_pool(), std::make_shared<arrow::StringType>());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      if (i % 5 == 0) {
        ASSERT_OK(builder.AppendNull());
        if (i < OUTPUT_SIZE) {
          buffer_8.indicator_pointer()[i] = SQL_NULL_DATA;
        } else {
          buffer_8_2.indicator_pointer()[i - OUTPUT_SIZE] = SQL_NULL_DATA;
        }
      } else {
        std::string str = std::to_string(i);
        ASSERT_OK(builder.Append(str));
        if (i < OUTPUT_SIZE) {
          memcpy(buffer_8[i].data_pointer, str.c_str(), str.size() + 1);
          buffer_8[i].indicator = str.size();
        } else {
          memcpy(buffer_8_2[i - OUTPUT_SIZE].data_pointer, str.c_str(), str.size() + 1);
          buffer_8_2[i - OUTPUT_SIZE].indicator = str.size();
        }
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[8], array));
  }
	cpp_odbc::multi_value_buffer buffer_9(sizeof(int64_t), OUTPUT_SIZE);
	cpp_odbc::multi_value_buffer buffer_9_2(sizeof(int64_t), OUTPUT_SIZE);
  {
    arrow::Int64Builder builder(arrow::default_memory_pool(), arrow::int64());
    for (int64_t i = 0; i < 2 * OUTPUT_SIZE; i++) {
      if (i % 5 == 0) {
        ASSERT_OK(builder.AppendNull());
        if (i < OUTPUT_SIZE) {
          buffer_9.indicator_pointer()[i] = SQL_NULL_DATA;
        } else {
          buffer_9_2.indicator_pointer()[i - OUTPUT_SIZE] = SQL_NULL_DATA;
        }
      } else {
        ASSERT_OK(builder.Append(i));
      }
    }
    std::shared_ptr<arrow::Array> array;
    ASSERT_OK(builder.Finish(&array));
    columns.emplace_back(std::make_shared<arrow::Column>(fields[9], array));
    auto typed_array = static_cast<arrow::Int64Array*>(array.get());
    memcpy(buffer_9.data_pointer(), typed_array->data()->data(), sizeof(int64_t) * OUTPUT_SIZE);
    memcpy(buffer_9_2.data_pointer(), typed_array->data()->data() + sizeof(int64_t) * OUTPUT_SIZE, sizeof(int64_t) * OUTPUT_SIZE);
  }
  std::shared_ptr<arrow::Table> expected_table = std::make_shared<arrow::Table>("", schema, columns);

	std::vector<turbodbc::column_info> expected = {
      {"float_column", turbodbc::type_code::floating_point, true},
      {"boolean_column", turbodbc::type_code::boolean, true},
      {"timestamp_column", turbodbc::type_code::timestamp, true},
      {"string_column", turbodbc::type_code::string, true},
      {"int_column", turbodbc::type_code::integer, true},
      {"nonnull_float_column", turbodbc::type_code::floating_point, false},
      {"nonnull_boolean_column", turbodbc::type_code::boolean, false},
      {"nonnull_timestamp_column", turbodbc::type_code::timestamp, false},
      {"nonnull_string_column", turbodbc::type_code::string, false},
      {"nonnull_int_column", turbodbc::type_code::integer, false}};
      // {"date_column", turbodbc::type_code::date, true},
      // {"nonnull_date_column", turbodbc::type_code::date, false}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  // Mock output columns
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers_1 = {
    buffer_0, buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, buffer_6, buffer_7, buffer_8, buffer_9};
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers_2 = {
    buffer_0_2, buffer_1_2, buffer_2_2, buffer_3_2, buffer_4_2, buffer_5_2, buffer_6_2, buffer_7_2, buffer_8_2, buffer_9_2};
	EXPECT_CALL(rs, do_get_buffers()).WillOnce(testing::Return(expected_buffers_1)).WillOnce(testing::Return(expected_buffers_2));
  EXPECT_CALL(rs, do_fetch_next_batch()).WillOnce(testing::Return(OUTPUT_SIZE)).WillOnce(testing::Return(OUTPUT_SIZE)).WillOnce(testing::Return(0));

  turbodbc_arrow::arrow_result_set ars(rs);
  std::shared_ptr<arrow::Table> table;
  ASSERT_OK(ars.fetch_all_native(&table));
  // TODO(ARROW-415): ASSERT_TRUE(expected_table->Equals(table));
  ASSERT_EQ(expected_table->name(), table->name());
  ASSERT_TRUE(expected_table->schema()->Equals(table->schema()));
  ASSERT_EQ(expected_table->num_columns(), table->num_columns());
  ASSERT_EQ(expected_table->num_rows(), table->num_rows());
  for (int i = 0; i < expected_table->num_columns(); i++) {
    // TODO(ARROW-416): ASSERT_TRUE(expected_table->column(i)->Equals(table->column(i)));
    ASSERT_EQ(expected_table->column(i)->length(), table->column(i)->length());
    ASSERT_EQ(expected_table->column(i)->null_count(), table->column(i)->null_count());
    ASSERT_TRUE(expected_table->column(i)->field()->Equals(table->column(i)->field()));
    std::shared_ptr<arrow::ChunkedArray> expected_c_array = expected_table->column(i)->data();
    std::shared_ptr<arrow::ChunkedArray> c_array = table->column(i)->data();
    // TODO(ARROW-417): ASSERT_TRUE(expected_c_array->Equals(c_array));
    // .. until then we assume a single chunk:
    ASSERT_EQ(c_array->num_chunks(), 1);
    ASSERT_TRUE(expected_c_array->chunk(0)->Equals(c_array->chunk(0)));
  }
}


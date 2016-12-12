#include <turbodbc_arrow/arrow_result_set.h>

#undef BOOL
#include <arrow/schema.h>
#include <arrow/test-util.h>
#include <arrow/types/primitive.h>
#include <arrow/util/memory-pool.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

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
  // !!! HERE !!!
  std::vector<std::shared_ptr<arrow::Field>> fields = {
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
  std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(fields);

  // TODO: Create arrow::Arrays
  // TODO: Build arrow::Table from arrays

  /*
  // Mock schema
  // TODO
	std::vector<turbodbc::column_info> expected = {{
    "int_column", turbodbc::type_code::integer, true}};
	EXPECT_CALL(rs, do_get_column_info()).WillRepeatedly(testing::Return(expected));

  // Mock output columns
  // * Two batches of 100 ints
	cpp_odbc::multi_value_buffer buffer_1(sizeof(int64_t), OUTPUT_SIZE);
  // TODO: add all arrays here
  memcpy(buffer_1.data_pointer(), typed_array->data()->data(), sizeof(int64_t) * OUTPUT_SIZE);
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> expected_buffers_1 = {buffer_1};
	cpp_odbc::multi_value_buffer buffer_2(sizeof(int64_t), OUTPUT_SIZE);
  // TODO: add all arrays here
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
  */
}


#include <turbodbc/result_set.h>

#include <gtest/gtest.h>

#include "cpp_odbc/connection.h"
#include "turbodbc/connection.h"
#include "mock_classes.h"

#include <sqlext.h>
#include <boost/variant/get.hpp>
#include <cstring>


using turbodbc_test::mock_connection;
using turbodbc_test::mock_statement;

namespace {

	std::shared_ptr<testing::NiceMock<mock_statement>> prepare_mock_with_columns(std::vector<SQLSMALLINT> const & column_types)
	{
		auto statement = std::make_shared<testing::NiceMock<mock_statement>>();

		ON_CALL(*statement, do_number_of_columns())
			.WillByDefault(testing::Return(column_types.size()));

		for (std::size_t i = 0; i != column_types.size(); ++i) {
			cpp_odbc::column_description const value = {"dummy_name", column_types[i], 42, 17, true};
			ON_CALL(*statement, do_describe_column(i + 1))
				.WillByDefault(testing::Return(value));
		}
		return statement;
	}

	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg2;
	}

	/**
	 * Set the mock statement to expect calls to bind_buffer. Returns a vector of pointers to buffers
	 * which are bound to the individual columns. These values will be filled when bind_column is called.
	 */
	std::vector<cpp_odbc::multi_value_buffer *> expect_calls_to_bind_buffer(mock_statement & statement, std::vector<SQLSMALLINT> const & expected_bind_types)
	{
		std::vector<cpp_odbc::multi_value_buffer *> buffers(expected_bind_types.size(), nullptr);

		for (std::size_t i = 0; i != expected_bind_types.size(); ++i) {
			EXPECT_CALL(statement, do_bind_column(i + 1, expected_bind_types[0], testing::_))
				.WillOnce(store_pointer_to_buffer_in(&buffers[i]));
		}

		return buffers;
	}

	/**
	* Change the address of the given target_pointer to point to the second argument of the mocked function
	*/
	ACTION_P(store_length_buffer_address_in, target_pointer) {
		*target_pointer = arg1;
	}

	void expect_rows_fetched_pointer_set(mock_statement & statement, SQLULEN * & rows_fetched)
	{
		EXPECT_CALL(statement, do_set_attribute(SQL_ATTR_ROWS_FETCHED_PTR, testing::An<SQLULEN *>()))
				.WillOnce(store_length_buffer_address_in(&rows_fetched));
	}

}


TEST(ResultSetTest, EnablesFetchingMultipleRows)
{
	auto statement = prepare_mock_with_columns({});
	std::size_t const buffered_rows = 42;

	EXPECT_CALL(*statement, do_set_attribute(SQL_ATTR_ROW_ARRAY_SIZE, buffered_rows)).Times(1);
	EXPECT_CALL(*statement, do_set_attribute(SQL_ATTR_ROWS_FETCHED_PTR, testing::An<SQLULEN *>())).Times(1);

	turbodbc::result_set result_set(statement, buffered_rows);
}

TEST(ResultSetTest, FetchWithoutColumns)
{
	auto statement = prepare_mock_with_columns({});

	ON_CALL(*statement, do_fetch_next())
		.WillByDefault(testing::Return(true));

	turbodbc::result_set result_set(statement, 1);
	ASSERT_EQ(0, result_set.fetch_one().size());
}


namespace {
	// store a value in a given pointer location
	ACTION_P2(set_rows_fetched, pointer_to_buffer, value) {
		*pointer_to_buffer = value;
	}
}

TEST(ResultSetTest, FetchEmptyColumn)
{
	auto statement = prepare_mock_with_columns({SQL_VARCHAR});
	auto buffers = expect_calls_to_bind_buffer(*statement, {SQL_CHAR});

	SQLULEN * rows_fetched = nullptr;
	expect_rows_fetched_pointer_set(*statement, rows_fetched);

	auto result_set = turbodbc::result_set(statement, 1);
	ASSERT_TRUE(buffers[0] != nullptr);

	EXPECT_CALL(*statement, do_fetch_next())
		.WillOnce(testing::DoAll(
					set_rows_fetched(rows_fetched, 0),
					testing::Return(false)
				));

	auto row = result_set.fetch_one();
	ASSERT_EQ(0, row.size());
}


namespace {
	/**
	* @brief Store the given string value as the first value in
	*        the buffer pointed to by pointer_to_buffer
	*/
	ACTION_P2(put_string_value_in_buffer, pointer_to_buffer, value) {
		auto element = (*pointer_to_buffer)[0];
		std::memcpy(element.data_pointer, value.data(), value.size() + 1);
		element.indicator = value.size();
	}
}

TEST(ResultSetTest, FetchWithSingleStringColumn)
{
	auto statement = prepare_mock_with_columns({SQL_VARCHAR});
	auto buffers = expect_calls_to_bind_buffer(*statement, {SQL_CHAR});

	SQLULEN * rows_fetched = nullptr;
	expect_rows_fetched_pointer_set(*statement, rows_fetched);

	auto result_set = turbodbc::result_set(statement, 1);
	ASSERT_TRUE(buffers[0] != nullptr);

	std::string const expected_value = "this is a test string";
	EXPECT_CALL(*statement, do_fetch_next())
		.WillOnce(testing::DoAll(
					put_string_value_in_buffer(buffers[0], expected_value),
					set_rows_fetched(rows_fetched, 1),
					testing::Return(true)
				));

	auto row = result_set.fetch_one();
	ASSERT_EQ(1, row.size());
	EXPECT_EQ(expected_value, boost::get<std::string>(*row[0]));
}


namespace {
	/**
	 * @brief Store the given binary value as the first value in
	 *        the buffer pointed to by pointer_to_buffer
	 */
	ACTION_P2(put_binary_value_in_buffer, pointer_to_buffer, value) {
		auto element = (*pointer_to_buffer)[0];
		std::memcpy(pointer_to_buffer->data_pointer(), &value, sizeof(value));
		element.indicator = sizeof(value);
	}

	/**
	 * @brief Store the given binary values as the first values in
	 *        the buffer pointed to by pointer_to_buffer
	 */
	ACTION_P2(put_binary_values_in_buffer, pointer_to_buffer, values) {
		for (std::size_t i = 0; i != values.size(); ++i) {
			auto element = (*pointer_to_buffer)[i];
			auto const element_size = sizeof(values[i]);
			std::memcpy(element.data_pointer, &values[i], element_size);
			element.indicator = element_size;
		}
	}
}

TEST(ResultSetTest, FetchWithSingleIntegerColumn)
{
	auto statement = prepare_mock_with_columns({SQL_INTEGER});
	auto buffers = expect_calls_to_bind_buffer(*statement, {SQL_C_SBIGINT});

	SQLULEN * rows_fetched = nullptr;
	expect_rows_fetched_pointer_set(*statement, rows_fetched);

	auto result_set = turbodbc::result_set(statement, 1);
	ASSERT_TRUE(buffers[0] != nullptr);

	long const expected_value = 42;
	EXPECT_CALL(*statement, do_fetch_next())
		.WillOnce(testing::DoAll(
					put_binary_value_in_buffer(buffers[0], expected_value),
					set_rows_fetched(rows_fetched, 1),
					testing::Return(true)
				));

	auto row = result_set.fetch_one();
	ASSERT_EQ(1, row.size());
	EXPECT_EQ(expected_value, boost::get<long>(*row[0]));
}


TEST(ResultSetTest, FetchWithMultipleColumns)
{
	auto statement = prepare_mock_with_columns({SQL_INTEGER, SQL_INTEGER});
	auto buffers = expect_calls_to_bind_buffer(*statement, {SQL_C_SBIGINT, SQL_C_SBIGINT});

	SQLULEN * rows_fetched = nullptr;
	expect_rows_fetched_pointer_set(*statement, rows_fetched);

	auto result_set = turbodbc::result_set(statement, 1);

	std::vector<long> expected_values = {42, 17};
	EXPECT_CALL(*statement, do_fetch_next())
		.WillOnce(testing::DoAll(
					put_binary_value_in_buffer(buffers[0], expected_values[0]),
					put_binary_value_in_buffer(buffers[1], expected_values[1]),
					set_rows_fetched(rows_fetched, 1),
					testing::Return(true)
				));

	auto row = result_set.fetch_one();
	ASSERT_EQ(2, row.size());
	EXPECT_EQ(expected_values[0], boost::get<long>(*row[0]));
	EXPECT_EQ(expected_values[1], boost::get<long>(*row[1]));
}


namespace {

	void test_fetch_multiple_rows(std::size_t rows, std::size_t buffer_size)
	{
		auto statement = prepare_mock_with_columns({SQL_INTEGER});
		auto buffers = expect_calls_to_bind_buffer(*statement, {SQL_C_SBIGINT});

		SQLULEN * rows_fetched = nullptr;
		expect_rows_fetched_pointer_set(*statement, rows_fetched);

		auto result_set = turbodbc::result_set(statement, buffer_size);

		std::vector<long> expected_values;
		for (std::size_t i = 0; i != rows; ++i) {
			expected_values.push_back(42 + i);
		}

		auto const complete_fetches = (rows / buffer_size);
		auto const remainder = rows % buffer_size;

		// prepare database api calls: first batches to fill the buffer, then
		// the last batch, then an empty batch
		testing::InSequence sequence;
		for (std::size_t i = 0; i != complete_fetches; ++i) {
			std::vector<long> fetched_data;
			for (std::size_t j = 0; j != buffer_size; ++j) {
				fetched_data.push_back(expected_values[i * buffer_size + j]);
			}

			EXPECT_CALL(*statement, do_fetch_next())
				.WillOnce(testing::DoAll(
							put_binary_values_in_buffer(buffers[0], fetched_data),
							set_rows_fetched(rows_fetched, fetched_data.size()),
							testing::Return(true)
						))
				.RetiresOnSaturation();
		}

		if (remainder > 0) {
			std::vector<long> fetched_data;
			for (std::size_t j = 0; j != remainder; ++j) {
				fetched_data.push_back(expected_values[complete_fetches * buffer_size + j]);
			}

			EXPECT_CALL(*statement, do_fetch_next())
				.WillOnce(testing::DoAll(
							put_binary_values_in_buffer(buffers[0], fetched_data),
							set_rows_fetched(rows_fetched, fetched_data.size()),
							testing::Return(true)
						))
				.RetiresOnSaturation();
		}

		EXPECT_CALL(*statement, do_fetch_next())
			.WillOnce(testing::DoAll(
						set_rows_fetched(rows_fetched, 0),
						testing::Return(false)
					));

		// assertions
		for (auto const & value : expected_values) {
			EXPECT_EQ( value, boost::get<long>(*result_set.fetch_one()[0]) );
		}
		EXPECT_TRUE( result_set.fetch_one().empty() );
	}


}

TEST(ResultSetTest, FetchRows2Buffer3)
{
	test_fetch_multiple_rows(2, 3);
}

TEST(ResultSetTest, FetchRows3Buffer3)
{
	test_fetch_multiple_rows(3, 3);
}

TEST(ResultSetTest, FetchRows4Buffer3)
{
	test_fetch_multiple_rows(4, 3);
}

TEST(ResultSetTest, FetchRows6Buffer3)
{
	test_fetch_multiple_rows(6, 3);
}

TEST(ResultSetTest, GetInfo)
{
	auto statement = prepare_mock_with_columns({SQL_VARCHAR, SQL_INTEGER});

	auto result_set = turbodbc::result_set(statement, 1);
	auto const columns = result_set.get_info();
	ASSERT_EQ(2, columns.size());
	EXPECT_EQ(turbodbc::type_code::string, columns[0].type);
	EXPECT_EQ(turbodbc::type_code::integer, columns[1].type);
}

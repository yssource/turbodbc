#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <tests/mock_classes.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <type_traits>
#include <algorithm>
#include <memory>

#include <sqlext.h>

using turbodbc::result_sets::double_buffered_result_set;
using turbodbc::column_info;
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
	 * Set the mock statement to expect calls to bind_buffer for count times.
	 */
	void expect_calls_to_bind_buffer(mock_statement & statement, std::vector<SQLSMALLINT> const & expected_bind_types, std::size_t count)
	{
		for (std::size_t i = 0; i != expected_bind_types.size(); ++i) {
			EXPECT_CALL(statement, do_bind_column(i + 1, expected_bind_types[i], testing::_)).Times(count);
		}
	}

}


TEST(DoubleBufferedResultSetTest, IsResultSet)
{
	bool const is_result_set = std::is_base_of<turbodbc::result_sets::result_set, double_buffered_result_set>::value;
	EXPECT_TRUE(is_result_set);
}


TEST(DoubleBufferedResultSetTest, BindsArraySizeInContructor)
{
	std::vector<SQLSMALLINT> const sql_column_types = {SQL_INTEGER, SQL_VARCHAR};
	std::vector<SQLSMALLINT> const c_column_types = {SQL_C_SBIGINT, SQL_CHAR};
	std::size_t const buffered_rows = 1001;

	auto statement = prepare_mock_with_columns(sql_column_types);
	EXPECT_CALL(*statement, do_set_attribute(SQL_ATTR_ROW_ARRAY_SIZE, 501));

	double_buffered_result_set rs(statement, buffered_rows);
}


namespace {

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

	ACTION_P2(set_pointer_to_value, target_pointer, new_value) {
		*target_pointer = new_value;
		return (new_value != 0);
	}

	void expect_fetch_next(mock_statement & statement, SQLULEN * rows_fetched_ptr, std::vector<std::size_t> values_to_set)
	{
		EXPECT_CALL(statement, do_fetch_next())
				.WillRepeatedly(set_pointer_to_value(rows_fetched_ptr, 0));

		std::reverse(values_to_set.begin(), values_to_set.end());
		for (auto value : values_to_set) {
			EXPECT_CALL(statement, do_fetch_next())
					.WillOnce(set_pointer_to_value(rows_fetched_ptr, value));
		}
	}

}


TEST(DoubleBufferedResultSetTest, FetchNextBatch)
{
	std::vector<SQLSMALLINT> const sql_column_types = {SQL_INTEGER};
	SQLULEN * rows_fetched = nullptr;
	auto statement = prepare_mock_with_columns(sql_column_types);
	expect_rows_fetched_pointer_set(*statement, rows_fetched);

	double_buffered_result_set rs(statement, 1000);
	ASSERT_TRUE(rows_fetched != nullptr);

	expect_fetch_next(*statement, rows_fetched, {123});
	EXPECT_EQ(123, rs.fetch_next_batch());
}


TEST(DoubleBufferedResultSetTest, GetColumnInfo)
{
	auto statement = prepare_mock_with_columns({SQL_INTEGER, SQL_VARCHAR});

	double_buffered_result_set rs(statement, 123);

	ASSERT_EQ(2, rs.get_column_info().size());
	EXPECT_EQ(turbodbc::type_code::integer, rs.get_column_info()[0].type);
	EXPECT_EQ(turbodbc::type_code::string, rs.get_column_info()[1].type);
}


//TEST(DoubleBufferedResultSetTest, GetBuffers)
//{
//	auto statement = prepare_mock_with_columns({SQL_INTEGER, SQL_VARCHAR});
//	std::size_t const buffered_rows = 1234;
//
//	double_buffered_result_set rs(statement, buffered_rows);
//	auto const buffers = rs.get_buffers();
//	ASSERT_EQ(2, buffers.size());
//
//	// make sure we can read the last elements for both columns
//	auto last_of_col_a = buffers[0].get()[buffered_rows - 1];
//	auto last_of_col_b = buffers[1].get()[buffered_rows - 1];
//
//	EXPECT_EQ(last_of_col_a.indicator, last_of_col_a.indicator);
//	EXPECT_EQ(last_of_col_b.indicator, last_of_col_b.indicator);
//}

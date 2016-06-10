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


TEST(DoubleBufferedResultSetTest, GetColumnInfo)
{
	auto statement = prepare_mock_with_columns({SQL_INTEGER, SQL_VARCHAR});

	double_buffered_result_set rs(statement, 123);

	ASSERT_EQ(2, rs.get_column_info().size());
	EXPECT_EQ(turbodbc::type_code::integer, rs.get_column_info()[0].type);
	EXPECT_EQ(turbodbc::type_code::string, rs.get_column_info()[1].type);
}


namespace {

	/**
	 * This class is a fake statement which implements functions relevant for
	 * the result set handling on top of a mock object.
	 */
	class statement_with_fake_int_result_set : public mock_statement {
	public:
		statement_with_fake_int_result_set(std::vector<size_t> batch_sizes) :
			rows_fetched_pointer_(nullptr),
			batch_sizes_(std::move(batch_sizes)),
			batch_index_(0)
		{}

		short int do_number_of_columns() const final
		{
			return 1;
		}

		cpp_odbc::column_description do_describe_column(SQLUSMALLINT) const final
		{
			return {"dummy_name", SQL_INTEGER, 42, 17, true};
		}

		void do_set_attribute(SQLINTEGER attribute, SQLULEN * pointer) const final
		{
			if (attribute == SQL_ATTR_ROWS_FETCHED_PTR) {
				rows_fetched_pointer_ = pointer;
			}
		};

		bool do_fetch_next() const final
		{
			if (batch_index_ < batch_sizes_.size()) {
				*rows_fetched_pointer_ = batch_sizes_[batch_index_];
				++batch_index_;
			} else {
				*rows_fetched_pointer_ = 0;
			}
			return (*rows_fetched_pointer_ != 0);
		};

	private:
		mutable SQLULEN * rows_fetched_pointer_;
		std::vector<size_t> batch_sizes_;
		mutable std::size_t batch_index_;
	};

}


TEST(DoubleBufferedResultSetTest, FetchNextBatch)
{
	std::vector<size_t> batch_sizes = {123};
	auto statement = std::make_shared<testing::NiceMock<statement_with_fake_int_result_set>>(batch_sizes);

	double_buffered_result_set rs(statement, 1000);
	EXPECT_EQ(123, rs.fetch_next_batch());
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

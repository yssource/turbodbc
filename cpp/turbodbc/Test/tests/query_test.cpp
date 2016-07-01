#include <turbodbc/query.h>

#include <turbodbc/result_sets/bound_result_set.h>
#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <gtest/gtest.h>
#include <cpp_odbc/connection.h>

#include "mock_classes.h"
#include <sqlext.h>


using turbodbc_test::mock_connection;
using turbodbc_test::mock_statement;


namespace {

	bool const no_double_buffering = false;
	bool const use_double_buffering = true;

}

TEST(QueryTest, GetRowCountBeforeExecuted)
{
	auto statement = std::make_shared<mock_statement>();

	turbodbc::query query(statement, 1, 1, no_double_buffering);
	EXPECT_EQ(0, query.get_row_count());
}

namespace {

	void prepare_single_column_result_set(mock_statement & statement)
	{
		ON_CALL( statement, do_number_of_columns())
				.WillByDefault(testing::Return(1));
		ON_CALL( statement, do_describe_column(testing::_))
				.WillByDefault(testing::Return(cpp_odbc::column_description{"", SQL_BIGINT, 8, 0, false}));
	}

}

TEST(QueryTest, GetRowCountAfterQueryWithResultSet)
{
	long const expected = 17;
	auto statement = std::make_shared<mock_statement>();

	prepare_single_column_result_set(*statement);
	EXPECT_CALL( *statement, do_row_count())
			.WillOnce(testing::Return(expected));

	turbodbc::query query(statement, 1, 1, no_double_buffering);
	query.execute();
	EXPECT_EQ(expected, query.get_row_count());
}


namespace {

	template <typename ExpectedResultSetType>
	void test_double_buffering(bool double_buffering)
	{
		auto statement = std::make_shared<mock_statement>();
		prepare_single_column_result_set(*statement);

		turbodbc::query query(statement, 1, 1, double_buffering);
		query.execute();
		EXPECT_TRUE(std::dynamic_pointer_cast<ExpectedResultSetType>(query.get_results()));
	}

}

TEST(QueryTest, UseDoubleBufferingAffectsResultSet)
{
	test_double_buffering<turbodbc::result_sets::bound_result_set>(no_double_buffering);
	test_double_buffering<turbodbc::result_sets::double_buffered_result_set>(use_double_buffering);
}

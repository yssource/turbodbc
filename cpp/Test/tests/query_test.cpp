#include "turbodbc/query.h"

#include <gtest/gtest.h>
#include "cpp_odbc/connection.h"

#include "mock_classes.h"
#include <sqlext.h>


using turbodbc_test::mock_connection;
using turbodbc_test::mock_statement;


TEST(QueryTest, FetchOneIfEmpty)
{
	auto statement = std::make_shared<mock_statement>();
	turbodbc::query query(statement, 1, 1);
	ASSERT_THROW(query.fetch_one(), std::runtime_error);
}


TEST(QueryTest, GetRowCountBeforeExecuted)
{
	auto statement = std::make_shared<mock_statement>();

	turbodbc::query query(statement, 1, 1);
	EXPECT_EQ(0, query.get_row_count());
}

TEST(QueryTest, GetRowCountAfterQueryWithResultSet)
{
	long const expected = 17;
	auto statement = std::make_shared<mock_statement>();

	// define a single column result set
	ON_CALL( *statement, do_number_of_columns())
			.WillByDefault(testing::Return(1));
	ON_CALL( *statement, do_describe_column(testing::_))
			.WillByDefault(testing::Return(cpp_odbc::column_description{"", SQL_BIGINT, 8, 0, false}));

	EXPECT_CALL( *statement, do_row_count())
			.WillOnce(testing::Return(expected));

	turbodbc::query query(statement, 1, 1);
	query.execute();
	EXPECT_EQ(expected, query.get_row_count());
}

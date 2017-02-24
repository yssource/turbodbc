#include <turbodbc/connection.h>

#include <gtest/gtest.h>
#include <cpp_odbc/connection.h>
#include "mock_classes.h"

#include <sqlext.h>


using turbodbc_test::mock_connection;
using turbodbc_test::mock_statement;

TEST(ConnectionTest, ConstructorDisablesAutoCommit)
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_set_attribute(SQL_ATTR_AUTOCOMMIT, SQL_AUTOCOMMIT_OFF)).Times(1);

	turbodbc::connection test_connection(connection);
}

TEST(ConnectionTest, Commit)
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_commit()).Times(1);

	turbodbc::connection test_connection(connection);
	test_connection.commit();
}

TEST(ConnectionTest, Rollback)
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_rollback()).Times(1);

	turbodbc::connection test_connection(connection);
	test_connection.rollback();
}

TEST(ConnectionTest, MakeCursorForwardsSelf)
{
	auto connection = std::make_shared<mock_connection>();

	turbodbc::connection test_connection(connection);
	auto cursor = test_connection.make_cursor();
	EXPECT_EQ(connection, cursor.get_connection());
}

TEST(ConnectionTest, BufferSizeDefault)
{
	auto connection = std::make_shared<mock_connection>();

	turbodbc::connection test_connection(connection);
	EXPECT_EQ(20, boost::get<turbodbc::megabytes>(test_connection.get_buffer_size()).value);
}

TEST(ConnectionTest, SetBufferSize)
{
	auto connection = std::make_shared<mock_connection>();

	turbodbc::connection test_connection(connection);
	test_connection.set_buffer_size(turbodbc::buffer_size(turbodbc::rows(999)));
	EXPECT_EQ(999, boost::get<turbodbc::rows>(test_connection.get_buffer_size()).value);
}

TEST(ConnectionTest, ParameterSetsToBufferDefault)
{
	auto connection = std::make_shared<mock_connection>();

	turbodbc::connection test_connection(connection);
	EXPECT_EQ(1000, test_connection.parameter_sets_to_buffer);
}

TEST(ConnectionTest, AsyncIODefault)
{
	auto connection = std::make_shared<mock_connection>();

	turbodbc::connection test_connection(connection);
	EXPECT_FALSE(test_connection.use_async_io);
}

TEST(ConnectionTest, SupportsDescribeParameter)
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_supports_function(SQL_API_SQLDESCRIBEPARAM))
		.WillOnce(testing::Return(false));

	turbodbc::connection test_connection(connection);
	EXPECT_FALSE(test_connection.supports_describe_parameter());
}


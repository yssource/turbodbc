#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include "cpp_odbc/connection.h"
#include "turbodbc/query.h"
#include "mock_classes.h"
#include <sqlext.h>


class query_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( query_test );

	CPPUNIT_TEST( fetch_one_if_empty );
	CPPUNIT_TEST( get_row_count_before_execute );
	CPPUNIT_TEST( get_row_count_after_query_with_result_set );


CPPUNIT_TEST_SUITE_END();

public:

	void fetch_one_if_empty();
	void get_row_count_before_execute();
	void get_row_count_after_query_with_result_set();


};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( query_test );

using turbodbc_test::mock_connection;
using turbodbc_test::mock_statement;


void query_test::fetch_one_if_empty()
{
	auto statement = std::make_shared<mock_statement>();
	turbodbc::query query(statement, 1, 1);
	CPPUNIT_ASSERT_THROW(query.fetch_one(), std::runtime_error);
}


void query_test::get_row_count_before_execute(){
	auto statement = std::make_shared<mock_statement>();

	turbodbc::query query(statement, 1, 1);
	CPPUNIT_ASSERT_EQUAL(0, query.get_row_count());
}

void query_test::get_row_count_after_query_with_result_set(){
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
	CPPUNIT_ASSERT_EQUAL(expected, query.get_row_count());
}

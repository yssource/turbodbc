#include <cppunit/extensions/HelperMacros.h>
#include "cpp_odbc/connection.h"
#include "pydbc/query.h"
#include "mock_classes.h"


class query_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( query_test );

	CPPUNIT_TEST( fetch_one_if_empty );
	CPPUNIT_TEST( get_row_count );


CPPUNIT_TEST_SUITE_END();

public:

	void fetch_one_if_empty();
	void get_row_count();


};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( query_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;


void query_test::fetch_one_if_empty()
{
	auto statement = std::make_shared<mock_statement>();
	pydbc::query query(statement);
	CPPUNIT_ASSERT_THROW(query.fetch_one(), std::runtime_error);
}


void query_test::get_row_count(){
	long const expected = 17;
	auto statement = std::make_shared<mock_statement>();
	EXPECT_CALL( *statement, do_row_count())
			.WillOnce(testing::Return(expected));

	pydbc::query query(statement);
	CPPUNIT_ASSERT_EQUAL(expected, query.get_row_count());
}

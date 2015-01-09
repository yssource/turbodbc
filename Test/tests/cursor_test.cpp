/**
 *  @file cursor_test.cpp
 *  @date 19.12.2014
 *  @author mwarsinsky
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */


#include <cppunit/extensions/HelperMacros.h>
#include "cpp_odbc/connection.h"
#include "pydbc/connection.h"
#include "mock_classes.h"


class cursor_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( cursor_test );

	CPPUNIT_TEST( get_statement );
	CPPUNIT_TEST( fetch_one_if_empty );
	CPPUNIT_TEST( get_rowcount );


CPPUNIT_TEST_SUITE_END();

public:

	void get_statement();
	void fetch_one_if_empty();
	void get_rowcount();


};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( cursor_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;

void cursor_test::get_statement()
{
	auto statement = std::make_shared<mock_statement const>();

	pydbc::cursor cursor(statement);
	CPPUNIT_ASSERT_EQUAL( statement, std::dynamic_pointer_cast<mock_statement const>(cursor.get_statement()));
}


void cursor_test::fetch_one_if_empty()
{
	auto statement = std::make_shared<mock_statement>();
	pydbc::cursor cursor(statement);
	CPPUNIT_ASSERT_THROW(cursor.fetch_one(), std::runtime_error);
}


void cursor_test::get_rowcount(){
	long const expected = 17;
	auto statement = std::make_shared<mock_statement>();
	EXPECT_CALL( *statement, do_row_count())
			.WillOnce(testing::Return(expected));

	pydbc::cursor cursor(statement);
	CPPUNIT_ASSERT_EQUAL(expected, cursor.get_rowcount());
}

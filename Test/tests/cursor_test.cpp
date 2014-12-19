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

	CPPUNIT_TEST( test_constructor );
	CPPUNIT_TEST( test_fetch_one_if_empty );
	CPPUNIT_TEST( test_get_rowcount_forwards );


CPPUNIT_TEST_SUITE_END();

public:

	void test_constructor();
	void test_fetch_one_if_empty();
	void test_get_rowcount_forwards();


};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( cursor_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;

void cursor_test::test_constructor()
{
	auto statement = std::make_shared<mock_statement>();

	CPPUNIT_ASSERT_NO_THROW(auto result=pydbc::cursor(statement));
	auto result_cursor=pydbc::cursor(statement);
	CPPUNIT_ASSERT_EQUAL( statement, std::dynamic_pointer_cast<mock_statement>(result_cursor.statement));

}
void cursor_test::test_fetch_one_if_empty()
{
	auto statement = std::make_shared<mock_statement>();
	auto result_cursor=pydbc::cursor(statement);
	CPPUNIT_ASSERT_THROW(result_cursor.fetch_one(), std::runtime_error);


}
void cursor_test::test_get_rowcount_forwards(){
	long const expected = 17;
	auto statement = std::make_shared<mock_statement>();
	auto result_cursor=pydbc::cursor(statement);
	EXPECT_CALL( *statement, do_row_count())
			.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL(expected, result_cursor.get_rowcount());

}

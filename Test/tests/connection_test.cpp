/**
 *  @file connection_test.cpp
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


class connection_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( connection_test );

	CPPUNIT_TEST( commit );
	CPPUNIT_TEST( test_make_cursor_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void commit();
	void test_make_cursor_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( connection_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;

void connection_test::commit()
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_commit()).Times(1);

	pydbc::connection test_connection(connection);
	test_connection.commit();
}

void connection_test::test_make_cursor_forwards()
{
	auto connection = std::make_shared<mock_connection>();
	auto statement = std::make_shared<mock_statement>();
	EXPECT_CALL(*connection, do_make_statement())
		.WillOnce(testing::Return(statement));

	pydbc::connection test_connection(connection);
	auto result_cursor = test_connection.make_cursor();
	CPPUNIT_ASSERT_EQUAL( statement, std::dynamic_pointer_cast<mock_statement>(result_cursor.statement));

}

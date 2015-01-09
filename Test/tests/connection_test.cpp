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

#include <sqlext.h>


class connection_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( connection_test );

	CPPUNIT_TEST( constructor_disables_auto_commit );
	CPPUNIT_TEST( commit );
	CPPUNIT_TEST( rollback );
	CPPUNIT_TEST( test_make_cursor_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void constructor_disables_auto_commit();
	void commit();
	void rollback();
	void test_make_cursor_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( connection_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;

void connection_test::constructor_disables_auto_commit()
{
	auto connection = std::make_shared<mock_connection>();
	EXPECT_CALL(*connection, do_set_attribute(SQL_ATTR_AUTOCOMMIT, SQL_AUTOCOMMIT_OFF)).Times(1);

	pydbc::connection test_connection(connection);
}


void connection_test::commit()
{
	auto connection = std::make_shared<testing::NiceMock<mock_connection>>();
	EXPECT_CALL(*connection, do_commit()).Times(1);

	pydbc::connection test_connection(connection);
	test_connection.commit();
}


void connection_test::rollback()
{
	auto connection = std::make_shared<testing::NiceMock<mock_connection>>();
	EXPECT_CALL(*connection, do_rollback()).Times(1);

	pydbc::connection test_connection(connection);
	test_connection.rollback();
}


void connection_test::test_make_cursor_forwards()
{
	auto connection = std::make_shared<testing::NiceMock<mock_connection>>();
	auto statement = std::make_shared<mock_statement const>();
	EXPECT_CALL(*connection, do_make_statement())
		.WillOnce(testing::Return(statement));

	pydbc::connection test_connection(connection);
	auto result_cursor = test_connection.make_cursor();
	CPPUNIT_ASSERT_EQUAL( statement, std::dynamic_pointer_cast<mock_statement const>(result_cursor.get_statement()));

}

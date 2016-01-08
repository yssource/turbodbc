#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <cpp_odbc/connection.h>
#include <pydbc/connection.h>
#include "mock_classes.h"

#include <sqlext.h>


class connection_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( connection_test );

	CPPUNIT_TEST( constructor_disables_auto_commit );
	CPPUNIT_TEST( commit );
	CPPUNIT_TEST( rollback );
	CPPUNIT_TEST( make_cursor_forwards_self );
	CPPUNIT_TEST( rows_to_buffer );

CPPUNIT_TEST_SUITE_END();

public:

	void constructor_disables_auto_commit();
	void commit();
	void rollback();
	void make_cursor_forwards_self();
	void rows_to_buffer();

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


void connection_test::make_cursor_forwards_self()
{
	auto connection = std::make_shared<testing::NiceMock<mock_connection>>();

	pydbc::connection test_connection(connection);
	auto cursor = test_connection.make_cursor();
	CPPUNIT_ASSERT(connection == cursor.get_connection());
}

void connection_test::rows_to_buffer()
{
	auto connection = std::make_shared<testing::NiceMock<mock_connection>>();

	pydbc::connection test_connection(connection);
	CPPUNIT_ASSERT_EQUAL(10, test_connection.rows_to_buffer);
}

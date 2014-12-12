/**
 *  @file connection_test.cpp
 *  @date 16.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc/connection.h"

#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/helpers/is_abstract_base_class.h"

#include "cpp_odbc_test/mock_statement.h"
#include "cpp_odbc_test/mock_connection.h"

class connection_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( connection_test );

	CPPUNIT_TEST( is_suitable_as_base_class );
	CPPUNIT_TEST( is_sharable );
	CPPUNIT_TEST( make_statement_forwards );
	CPPUNIT_TEST( set_integer_connection_attribute_forwards );
	CPPUNIT_TEST( commit_forwards );
	CPPUNIT_TEST( rollback_forwards );
	CPPUNIT_TEST( get_string_info_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void is_suitable_as_base_class();
	void is_sharable();
	void make_statement_forwards();
	void set_integer_connection_attribute_forwards();
	void commit_forwards();
	void rollback_forwards();
	void get_string_info_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( connection_test );

using cpp_odbc_test::mock_connection;

void connection_test::is_suitable_as_base_class()
{
	bool const is_base = cppunit_toolbox::is_abstract_base_class<cpp_odbc::connection>::value;
	CPPUNIT_ASSERT( is_base );
}

void connection_test::is_sharable()
{
	auto connection = std::make_shared<mock_connection>();
	auto shared = connection->shared_from_this();
	CPPUNIT_ASSERT( connection == shared );
}

void connection_test::make_statement_forwards()
{
	mock_connection connection;
	auto statement = std::make_shared<cpp_odbc_test::mock_statement>();

	EXPECT_CALL(connection, do_make_statement())
		.WillOnce(testing::Return(statement));

	CPPUNIT_ASSERT( statement == connection.make_statement() );
}

void connection_test::set_integer_connection_attribute_forwards()
{
	mock_connection connection;
	SQLINTEGER const attribute = 42;
	long const value = 17;

	EXPECT_CALL(connection, do_set_connection_attribute(attribute, value)).Times(1);

	connection.set_connection_attribute(attribute, value);
}

void connection_test::commit_forwards()
{
	mock_connection connection;

	EXPECT_CALL(connection, do_commit()).Times(1);

	connection.commit();
}

void connection_test::rollback_forwards()
{
	mock_connection connection;

	EXPECT_CALL(connection, do_rollback()).Times(1);

	connection.rollback();
}

void connection_test::get_string_info_forwards()
{
	mock_connection connection;
	SQLUSMALLINT const info_type = 42;
	std::string const expected = "test info";

	EXPECT_CALL(connection, do_get_string_info(info_type))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL(expected, connection.get_string_info(info_type));
}

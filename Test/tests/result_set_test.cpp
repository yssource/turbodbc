/**
 *  @file result_set_test.cpp
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
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include "cpp_odbc/connection.h"
#include "pydbc/connection.h"
#include "mock_classes.h"
#include "pydbc/result_set.h"
#include <sqlext.h>


class result_set_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( result_set_test );

	CPPUNIT_TEST( test_constructor_empty );
	CPPUNIT_TEST( test_constructor_one_string_type );
	CPPUNIT_TEST( test_constructor_one_non_string_type );
	CPPUNIT_TEST( test_constructor_both_types );

CPPUNIT_TEST_SUITE_END();

public:

	void test_constructor_empty();
	void test_constructor_one_string_type();
	void test_constructor_one_non_string_type();
	void test_constructor_both_types();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( result_set_test );

using pydbc_test::mock_connection;
using pydbc_test::mock_statement;

void result_set_test::test_constructor_empty()
{
	auto statement = std::make_shared<mock_statement>();

	EXPECT_CALL(*statement, do_number_of_columns()).WillOnce(testing::Return(0));
	auto result_set = pydbc::result_set(statement);

	CPPUNIT_ASSERT_EQUAL(0, result_set.columns.size());
}


void result_set_test::test_constructor_one_string_type()
{
	auto statement = std::make_shared<mock_statement>();

	EXPECT_CALL(*statement, do_number_of_columns())
		.WillOnce(testing::Return(1));
	EXPECT_CALL(*statement, do_get_integer_column_attribute(1, SQL_DESC_TYPE))
		.WillOnce(testing::Return(SQL_VARCHAR));
	EXPECT_CALL(*statement, do_bind_column(1, SQL_CHAR, testing::_)).Times(1);

	auto result_set = pydbc::result_set(statement);
	CPPUNIT_ASSERT_EQUAL(1, result_set.columns.size());
}


void result_set_test::test_constructor_one_non_string_type(){

	auto statement = std::make_shared<mock_statement>();

	EXPECT_CALL(*statement, do_number_of_columns())
		.WillOnce(testing::Return(1));
	EXPECT_CALL(*statement, do_get_integer_column_attribute(1, SQL_DESC_TYPE))
		.WillOnce(testing::Return(SQL_INTEGER));
	EXPECT_CALL(*statement, do_bind_column(1, SQL_C_SBIGINT, testing::_))
		.Times(1);

	auto result_set = pydbc::result_set(statement);
	CPPUNIT_ASSERT_EQUAL(1, result_set.columns.size());
}


void result_set_test::test_constructor_both_types()
{
	auto statement = std::make_shared<mock_statement>();

	EXPECT_CALL(*statement, do_number_of_columns())
		.WillOnce(testing::Return(2));
	EXPECT_CALL(*statement, do_get_integer_column_attribute(1,SQL_DESC_TYPE))
		.WillOnce(testing::Return(SQL_VARCHAR));
	EXPECT_CALL(*statement, do_get_integer_column_attribute(2,SQL_DESC_TYPE))
		.WillOnce(testing::Return(SQL_INTEGER));
	EXPECT_CALL(*statement, do_bind_column(1, SQL_CHAR, testing::_)).Times(1);
	EXPECT_CALL(*statement, do_bind_column(2, SQL_C_SBIGINT, testing::_)).Times(1);

	auto result_set = pydbc::result_set(statement);
	CPPUNIT_ASSERT_EQUAL(2, result_set.columns.size());
}


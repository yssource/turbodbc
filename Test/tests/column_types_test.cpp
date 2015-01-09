/**
 *  @file column_types_test.cpp
 *  @date 09.01.2015
 *  @author mkoenig
 */

#include "pydbc/column_types.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/helpers/is_abstract_base_class.h>

#include "mock_classes.h"
#include <sqlext.h>


class column_types_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_types_test );

	CPPUNIT_TEST( long_column_bound_on_construction );
	CPPUNIT_TEST( string_column_bound_on_construction );

CPPUNIT_TEST_SUITE_END();

public:

	void long_column_bound_on_construction();
	void string_column_bound_on_construction();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_types_test );

using pydbc_test::mock_statement;

void column_types_test::long_column_bound_on_construction()
{
	std::size_t const column_index = 42;
	auto const buffer_type = SQL_C_SBIGINT;
	mock_statement statement;

	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_)).Times(1);
	pydbc::long_column column(statement, column_index);
}

void column_types_test::string_column_bound_on_construction()
{
	std::size_t const column_index = 42;
	auto const buffer_type = SQL_CHAR;
	mock_statement statement;

	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_)).Times(1);
	pydbc::string_column column(statement, column_index);
}


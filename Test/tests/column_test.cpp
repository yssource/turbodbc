/**
 *  @file column_test.cpp
 *  @date 09.01.2015
 *  @author mkoenig
 */

#include "pydbc/column.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/helpers/is_abstract_base_class.h>

#include <gmock/gmock.h>


class column_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_test );

	CPPUNIT_TEST( is_base_class );
	CPPUNIT_TEST( get_field_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void is_base_class();
	void get_field_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_test );

namespace {

	struct mock_column : public pydbc::column {
		MOCK_CONST_METHOD0(do_get_field, pydbc::field());
	};

}

void column_test::is_base_class()
{
	bool const all_good = cppunit_toolbox::is_abstract_base_class<pydbc::column>::value;
	CPPUNIT_ASSERT(all_good);
}

void column_test::get_field_forwards()
{
	pydbc::field const expected = 42;

	mock_column column;
	EXPECT_CALL(column, do_get_field())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == column.get_field());
}

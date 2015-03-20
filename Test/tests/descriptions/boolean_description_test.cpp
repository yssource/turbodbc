#include "pydbc/descriptions/boolean_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class boolean_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( boolean_description_test );

	CPPUNIT_TEST( test );

CPPUNIT_TEST_SUITE_END();

public:

	void test();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( boolean_description_test );

void boolean_description_test::test()
{
	pydbc::boolean_description const description;

	CPPUNIT_ASSERT_EQUAL(1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_BIT, description.column_type());

	char as_bool = 0;
	CPPUNIT_ASSERT_EQUAL(pydbc::field{false}, description.make_field(&as_bool));
	as_bool = 1;
	CPPUNIT_ASSERT_EQUAL(pydbc::field{true}, description.make_field(&as_bool));
}

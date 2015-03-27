#include "pydbc/descriptions/floating_point_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class floating_point_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( floating_point_description_test );

	CPPUNIT_TEST( test );

CPPUNIT_TEST_SUITE_END();

public:

	void test();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( floating_point_description_test );

void floating_point_description_test::test()
{
	double const expected = 3.14;
	pydbc::floating_point_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(expected), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_DOUBLE, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_DOUBLE, description.column_sql_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

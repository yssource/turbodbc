#include "pydbc/descriptions/integer_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class integer_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( integer_description_test );

	CPPUNIT_TEST( test );

CPPUNIT_TEST_SUITE_END();

public:

	void test();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( integer_description_test );

void integer_description_test::test()
{
	long const expected = 42;
	pydbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(expected), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_SBIGINT, description.column_type());
	CPPUNIT_ASSERT_EQUAL(SQL_BIGINT, description.column_sql_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

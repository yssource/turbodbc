#include "pydbc/descriptions/string_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class string_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( string_description_test );

	CPPUNIT_TEST( test );

CPPUNIT_TEST_SUITE_END();

public:

	void test();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( string_description_test );

void string_description_test::test()
{
	std::size_t const size = 42;
	std::string const expected("test string");
	pydbc::string_description const description(size);

	CPPUNIT_ASSERT_EQUAL(size + 1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_CHAR, description.column_type());
	CPPUNIT_ASSERT_EQUAL(SQL_VARCHAR, description.column_sql_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(expected.c_str()));
}

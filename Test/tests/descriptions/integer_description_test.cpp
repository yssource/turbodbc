#include "pydbc/descriptions/integer_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class integer_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( integer_description_test );

	CPPUNIT_TEST( basic_properties );
	CPPUNIT_TEST( make_field );
	CPPUNIT_TEST( set_field );

CPPUNIT_TEST_SUITE_END();

public:

	void basic_properties();
	void make_field();
	void set_field();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( integer_description_test );

void integer_description_test::basic_properties()
{
	pydbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(long), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_SBIGINT, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_BIGINT, description.column_sql_type());
}

void integer_description_test::make_field()
{
	long const expected = 42;
	pydbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

void integer_description_test::set_field()
{
	long const expected = 42;
	pydbc::integer_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, pydbc::field{expected});
	CPPUNIT_ASSERT_EQUAL(expected, *reinterpret_cast<long *>(element.data_pointer));
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

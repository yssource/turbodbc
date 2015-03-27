#include "pydbc/descriptions/floating_point_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class floating_point_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( floating_point_description_test );

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
CPPUNIT_TEST_SUITE_REGISTRATION( floating_point_description_test );

void floating_point_description_test::basic_properties()
{
	pydbc::floating_point_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(double), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_DOUBLE, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_DOUBLE, description.column_sql_type());
}

void floating_point_description_test::make_field()
{
	double const expected = 3.14;
	pydbc::floating_point_description const description;

	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

void floating_point_description_test::set_field()
{
	double const expected = 42;
	pydbc::floating_point_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, pydbc::field{expected});
	CPPUNIT_ASSERT_EQUAL(expected, *reinterpret_cast<double *>(element.data_pointer));
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

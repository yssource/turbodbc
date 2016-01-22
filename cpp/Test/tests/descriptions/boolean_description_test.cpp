#include "turbodbc/descriptions/boolean_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class boolean_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( boolean_description_test );

	CPPUNIT_TEST( basic_properties );
	CPPUNIT_TEST( make_field );
	CPPUNIT_TEST( set_field );
	CPPUNIT_TEST( get_type_code );
	CPPUNIT_TEST( custom_name_and_nullable_support );

CPPUNIT_TEST_SUITE_END();

public:

	void basic_properties();
	void make_field();
	void set_field();
	void get_type_code();
	void custom_name_and_nullable_support();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( boolean_description_test );

void boolean_description_test::basic_properties()
{
	turbodbc::boolean_description const description;
	CPPUNIT_ASSERT_EQUAL(1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_BIT, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_BIT, description.column_sql_type());
}

void boolean_description_test::make_field()
{
	turbodbc::boolean_description const description;

	char as_bool = 0;
	CPPUNIT_ASSERT_EQUAL(turbodbc::field{false}, description.make_field(&as_bool));
	as_bool = 1;
	CPPUNIT_ASSERT_EQUAL(turbodbc::field{true}, description.make_field(&as_bool));
}

void boolean_description_test::set_field()
{
	turbodbc::boolean_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{true});
	CPPUNIT_ASSERT_EQUAL(1, *element.data_pointer);
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);

	description.set_field(element, turbodbc::field{false});
	CPPUNIT_ASSERT_EQUAL(0, *element.data_pointer);
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

void boolean_description_test::get_type_code()
{
	turbodbc::boolean_description const description;
	CPPUNIT_ASSERT( turbodbc::type_code::boolean == description.get_type_code() );
}

void boolean_description_test::custom_name_and_nullable_support()
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::boolean_description const description(expected_name, expected_supports_null);

	CPPUNIT_ASSERT_EQUAL(expected_name, description.name());
	CPPUNIT_ASSERT_EQUAL(expected_supports_null, description.supports_null_values());
}

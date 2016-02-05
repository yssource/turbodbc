#include "turbodbc/descriptions/string_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class string_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( string_description_test );

	CPPUNIT_TEST( basic_properties );
	CPPUNIT_TEST( make_field );
	CPPUNIT_TEST( set_field );
	CPPUNIT_TEST( set_field_with_maximum_length );
	CPPUNIT_TEST( set_field_raises_for_too_long_values );
	CPPUNIT_TEST( get_type_code );
	CPPUNIT_TEST( custom_name_and_nullable_support );

CPPUNIT_TEST_SUITE_END();

public:

	void basic_properties();
	void make_field();
	void set_field();
	void set_field_with_maximum_length();
	void set_field_raises_for_too_long_values();
	void get_type_code();
	void custom_name_and_nullable_support();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( string_description_test );

void string_description_test::basic_properties()
{
	std::size_t const size = 42;
	turbodbc::string_description const description(size);

	CPPUNIT_ASSERT_EQUAL(size + 1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_CHAR, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_VARCHAR, description.column_sql_type());
}

void string_description_test::make_field()
{
	std::string const expected("test string");
	turbodbc::string_description const description(42);

	CPPUNIT_ASSERT_EQUAL(turbodbc::field{expected}, description.make_field(expected.c_str()));
}

void string_description_test::set_field()
{
	std::string const expected("another test string");
	turbodbc::string_description const description(expected.size() + 1); // one larger

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{expected});
	CPPUNIT_ASSERT_EQUAL(expected, std::string(element.data_pointer));
	CPPUNIT_ASSERT_EQUAL(expected.size(), element.indicator);
}

void string_description_test::set_field_with_maximum_length()
{
	std::string const expected("another test string");
	turbodbc::string_description const description(expected.size());

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{expected});
	CPPUNIT_ASSERT_EQUAL(expected, std::string(element.data_pointer));
	CPPUNIT_ASSERT_EQUAL(expected.size(), element.indicator);
}

void string_description_test::set_field_raises_for_too_long_values()
{
	std::string const basic("another test string");
	std::string const full(basic + "x");
	turbodbc::string_description const description(basic.size());

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	CPPUNIT_ASSERT_THROW(description.set_field(element, turbodbc::field{full}), std::runtime_error);
}

void string_description_test::get_type_code()
{
	turbodbc::string_description const description(10);
	CPPUNIT_ASSERT( turbodbc::type_code::string == description.get_type_code() );
}

void string_description_test::custom_name_and_nullable_support()
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::string_description const description(expected_name, expected_supports_null, 10);

	CPPUNIT_ASSERT_EQUAL(expected_name, description.name());
	CPPUNIT_ASSERT_EQUAL(expected_supports_null, description.supports_null_values());
}

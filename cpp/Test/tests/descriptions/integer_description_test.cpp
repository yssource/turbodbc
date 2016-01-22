#include "turbodbc/descriptions/integer_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class integer_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( integer_description_test );

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
CPPUNIT_TEST_SUITE_REGISTRATION( integer_description_test );

void integer_description_test::basic_properties()
{
	turbodbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(long), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_SBIGINT, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_BIGINT, description.column_sql_type());
}

void integer_description_test::make_field()
{
	long const expected = 42;
	turbodbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(turbodbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

void integer_description_test::set_field()
{
	long const expected = 42;
	turbodbc::integer_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{expected});
	CPPUNIT_ASSERT_EQUAL(expected, *reinterpret_cast<long *>(element.data_pointer));
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

void integer_description_test::get_type_code()
{
	turbodbc::integer_description const description;
	CPPUNIT_ASSERT( turbodbc::type_code::integer == description.get_type_code() );
}

void integer_description_test::custom_name_and_nullable_support()
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::integer_description const description(expected_name, expected_supports_null);

	CPPUNIT_ASSERT_EQUAL(expected_name, description.name());
	CPPUNIT_ASSERT_EQUAL(expected_supports_null, description.supports_null_values());
}

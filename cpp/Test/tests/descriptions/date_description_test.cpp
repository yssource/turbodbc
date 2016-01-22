#include "turbodbc/descriptions/date_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class date_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( date_description_test );

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
CPPUNIT_TEST_SUITE_REGISTRATION( date_description_test );

void date_description_test::basic_properties()
{
	turbodbc::date_description const description;

	CPPUNIT_ASSERT_EQUAL(6, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_TYPE_DATE, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_TYPE_DATE, description.column_sql_type());
}

void date_description_test::make_field()
{
	boost::gregorian::date const expected{2015, 12, 31};
	turbodbc::date_description const description;

	SQL_DATE_STRUCT const sql_date = {2015, 12, 31};
	CPPUNIT_ASSERT_EQUAL(turbodbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&sql_date)));
}

void date_description_test::set_field()
{
	boost::gregorian::date const date{2015, 12, 31};
	turbodbc::date_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{date});
	auto const as_sql_date = reinterpret_cast<SQL_DATE_STRUCT const *>(element.data_pointer);
	CPPUNIT_ASSERT_EQUAL(2015, as_sql_date->year);
	CPPUNIT_ASSERT_EQUAL(12, as_sql_date->month);
	CPPUNIT_ASSERT_EQUAL(31, as_sql_date->day);
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

void date_description_test::get_type_code()
{
	turbodbc::date_description const description;
	CPPUNIT_ASSERT( turbodbc::type_code::date == description.get_type_code() );
}

void date_description_test::custom_name_and_nullable_support()
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::date_description const description(expected_name, expected_supports_null);

	CPPUNIT_ASSERT_EQUAL(expected_name, description.name());
	CPPUNIT_ASSERT_EQUAL(expected_supports_null, description.supports_null_values());
}

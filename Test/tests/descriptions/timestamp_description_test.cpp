#include "pydbc/descriptions/timestamp_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class timestamp_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( timestamp_description_test );

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
CPPUNIT_TEST_SUITE_REGISTRATION( timestamp_description_test );

void timestamp_description_test::basic_properties()
{
	pydbc::timestamp_description const description;

	CPPUNIT_ASSERT_EQUAL(16, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_TYPE_TIMESTAMP, description.column_c_type());
	CPPUNIT_ASSERT_EQUAL(SQL_TYPE_TIMESTAMP, description.column_sql_type());
}

void timestamp_description_test::make_field()
{
	boost::posix_time::ptime const expected{{2015, 12, 31}, {1, 2, 3, 123456}};
	pydbc::timestamp_description const description;

	SQL_TIMESTAMP_STRUCT const sql_timestamp = {2015, 12, 31, 1, 2, 3, 123456000};
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&sql_timestamp)));
}

void timestamp_description_test::set_field()
{
	boost::posix_time::ptime const timestamp{{2015, 12, 31}, {1, 2, 3, 123456}};
	pydbc::timestamp_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, pydbc::field{timestamp});
	auto const as_sql_date = reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(element.data_pointer);
	CPPUNIT_ASSERT_EQUAL(2015, as_sql_date->year);
	CPPUNIT_ASSERT_EQUAL(12, as_sql_date->month);
	CPPUNIT_ASSERT_EQUAL(31, as_sql_date->day);
	CPPUNIT_ASSERT_EQUAL(1, as_sql_date->hour);
	CPPUNIT_ASSERT_EQUAL(2, as_sql_date->minute);
	CPPUNIT_ASSERT_EQUAL(3, as_sql_date->second);
	CPPUNIT_ASSERT_EQUAL(123456000, as_sql_date->fraction);
	CPPUNIT_ASSERT_EQUAL(description.element_size(), element.indicator);
}

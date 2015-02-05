#include "cpp_odbc/column_description.h"

#include <cppunit/extensions/HelperMacros.h>


class column_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_description_test );

	CPPUNIT_TEST( members );

CPPUNIT_TEST_SUITE_END();

public:

	void members()
	{
		std::string const expected_name("dummy");
		SQLSMALLINT const expected_data_type = 42;
		SQLULEN const expected_size = 17;
		SQLSMALLINT const expected_decimal_digits = 3;
		bool const expected_allows_nullable = false;

		cpp_odbc::column_description const description = {
			expected_name,
			expected_data_type,
			expected_size,
			expected_decimal_digits,
			expected_allows_nullable
		};

		CPPUNIT_ASSERT_EQUAL(expected_name, description.name);
		CPPUNIT_ASSERT_EQUAL(expected_data_type, description.data_type);
		CPPUNIT_ASSERT_EQUAL(expected_size, description.size);
		CPPUNIT_ASSERT_EQUAL(expected_decimal_digits, description.decimal_digits);
		CPPUNIT_ASSERT_EQUAL(expected_allows_nullable, description.allows_null_values);
	}

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_description_test );

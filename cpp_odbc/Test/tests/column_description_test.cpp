#include "cpp_odbc/column_description.h"

#include <cppunit/extensions/HelperMacros.h>

#include <sstream>
#include <sqlext.h>

using cpp_odbc::column_description;

class column_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_description_test );

	CPPUNIT_TEST( members );
	CPPUNIT_TEST( equality );
	CPPUNIT_TEST( output_known_type );
	CPPUNIT_TEST( output_unknown_type );

CPPUNIT_TEST_SUITE_END();

public:

	void members()
	{
		std::string const expected_name("dummy");
		SQLSMALLINT const expected_data_type = 42;
		SQLULEN const expected_size = 17;
		SQLSMALLINT const expected_decimal_digits = 3;
		bool const expected_allows_nullable = false;

		column_description const description = {
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

	void equality()
	{
		column_description const original = {"dummy", 1, 2, 3, false};
		CPPUNIT_ASSERT( original == original );

		column_description different_name(original);
		different_name.name = "other";
		CPPUNIT_ASSERT( not (original == different_name) );

		column_description different_type(original);
		different_type.data_type += 1;
		CPPUNIT_ASSERT( not (original == different_type) );

		column_description different_size(original);
		different_size.size += 1;
		CPPUNIT_ASSERT( not (original == different_size) );

		column_description different_digits(original);
		different_digits.decimal_digits += 1;
		CPPUNIT_ASSERT( not (original == different_digits) );

		column_description different_null(original);
		different_null.allows_null_values = not original.allows_null_values;
		CPPUNIT_ASSERT( not (original == different_null) );
	}

	void test_output(std::string const & expected, column_description const & description)
	{
		std::ostringstream actual;
		actual << description;
		CPPUNIT_ASSERT_EQUAL(expected, actual.str());
	}

	void output_known_type()
	{
		test_output(
				"test_name @ SQL_INTEGER (precision 2, scale 3)",
				{"test_name", SQL_INTEGER, 2, 3, false}
			);

		test_output(
				"test_name @ NULLABLE SQL_INTEGER (precision 2, scale 3)",
				{"test_name", SQL_INTEGER, 2, 3, true}
			);
	}

	void output_unknown_type()
	{
		test_output(
				"test_name @ UNKNOWN TYPE (precision 2, scale 3)",
				{"test_name", 666, 2, 3, false}
			);
	}

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_description_test );

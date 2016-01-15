#include <pydbc/make_description.h>

#include <cppunit/extensions/HelperMacros.h>

#include <pydbc/descriptions.h>

#include <sqlext.h>
#include <sstream>
#include <stdexcept>


class make_description_of_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( make_description_of_description_test );

	CPPUNIT_TEST( unsupported_type_throws );
	CPPUNIT_TEST( integer_types );
	CPPUNIT_TEST( string_types );
	CPPUNIT_TEST( floating_point_types );
	CPPUNIT_TEST( bit_type );
	CPPUNIT_TEST( date_type );
	CPPUNIT_TEST( timestamp_type );
	CPPUNIT_TEST( decimal_as_integer );
	CPPUNIT_TEST( decimal_as_floating_point );
	CPPUNIT_TEST( decimal_as_string );

CPPUNIT_TEST_SUITE_END();

public:

	void unsupported_type_throws();
	void integer_types();
	void string_types();
	void floating_point_types();
	void bit_type();
	void date_type();
	void timestamp_type();
	void decimal_as_integer();
	void decimal_as_floating_point();
	void decimal_as_string();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( make_description_of_description_test );

using pydbc::make_description;

namespace {

	std::string const name("custom_name");
	bool const supports_null_values = false;

	void assert_custom_name_and_nullable_support(pydbc::description const & description)
	{
		CPPUNIT_ASSERT_EQUAL(name, description.name());
		CPPUNIT_ASSERT_EQUAL(supports_null_values, description.supports_null_values());
	}

	void test_as_integer(cpp_odbc::column_description const & column_description)
	{
		auto const description = make_description(column_description);
		std::ostringstream message;
		message << "Could not convert type identifier '" << column_description.data_type << "' to integer description";
		CPPUNIT_ASSERT_MESSAGE( message.str(), dynamic_cast<pydbc::integer_description const *>(description.get()) );

		assert_custom_name_and_nullable_support(*description);
	}

	void test_as_floating_point(cpp_odbc::column_description const & column_description)
	{
		auto const description = make_description(column_description);
		std::ostringstream message;
		message << "Could not convert type identifier '" << column_description.data_type << "' to floating point description";
		CPPUNIT_ASSERT_MESSAGE( message.str(), dynamic_cast<pydbc::floating_point_description const *>(description.get()) );

		assert_custom_name_and_nullable_support(*description);
	}

	void test_unsupported(cpp_odbc::column_description const & column_description)
	{
		CPPUNIT_ASSERT_THROW( make_description(column_description), std::runtime_error );
	}

	void test_as_string(cpp_odbc::column_description const & column_description, std::size_t expected_size)
	{
		auto const description = make_description(column_description);

		std::ostringstream message;
		message << "Could not convert type identifier '" << column_description.data_type << "' to string description";
		CPPUNIT_ASSERT_MESSAGE( message.str(), dynamic_cast<pydbc::string_description const *>(description.get()) );

		CPPUNIT_ASSERT_EQUAL( expected_size, description->element_size() );
		assert_custom_name_and_nullable_support(*description);
	}

}

void make_description_of_description_test::unsupported_type_throws()
{
	SQLSMALLINT const unsupported_type = SQL_GUID;
	cpp_odbc::column_description column_description = {name, unsupported_type, 0, 0, supports_null_values};
	test_unsupported(column_description);
}

void make_description_of_description_test::integer_types()
{
	std::vector<SQLSMALLINT> const types = {
			SQL_SMALLINT, SQL_INTEGER, SQL_TINYINT, SQL_BIGINT,
		};

	for (auto const type : types) {
		cpp_odbc::column_description column_description = {name, type, 0, 0, supports_null_values};
		test_as_integer(column_description);
	}
}

void make_description_of_description_test::string_types()
{
	std::vector<SQLSMALLINT> const types = {
			SQL_CHAR, SQL_VARCHAR, SQL_LONGVARCHAR, SQL_WCHAR, SQL_WVARCHAR, SQL_WLONGVARCHAR
		};

	std::size_t const size = 42;
	for (auto const type : types) {
		cpp_odbc::column_description column_description = {name, type, size, 0, supports_null_values};
		test_as_string(column_description, size + 1);
	}
}

void make_description_of_description_test::floating_point_types()
{
	std::vector<SQLSMALLINT> const types = {
			SQL_REAL, SQL_FLOAT, SQL_DOUBLE
		};

	for (auto const type : types) {
		cpp_odbc::column_description column_description = {name, type, 0, 0, supports_null_values};
		test_as_floating_point(column_description);
	}
}

void make_description_of_description_test::bit_type()
{
	SQLSMALLINT const type = SQL_BIT;

	cpp_odbc::column_description column_description = {name, type, 0, 0, supports_null_values};
	auto const description = make_description(column_description);
	CPPUNIT_ASSERT( dynamic_cast<pydbc::boolean_description const *>(description.get()) );
	assert_custom_name_and_nullable_support(*description);
}

void make_description_of_description_test::date_type()
{
	SQLSMALLINT const type = SQL_TYPE_DATE;

	cpp_odbc::column_description column_description = {name, type, 0, 0, supports_null_values};
	auto const description = make_description(column_description);
	CPPUNIT_ASSERT( dynamic_cast<pydbc::date_description const *>(description.get()) );
	assert_custom_name_and_nullable_support(*description);
}

void make_description_of_description_test::timestamp_type()
{
	SQLSMALLINT const type = SQL_TYPE_TIMESTAMP;

	cpp_odbc::column_description column_description = {name, type, 0, 0, supports_null_values};
	auto const description = make_description(column_description);
	CPPUNIT_ASSERT( dynamic_cast<pydbc::timestamp_description const *>(description.get()) );
	assert_custom_name_and_nullable_support(*description);
}

namespace {

	cpp_odbc::column_description make_decimal_column_description(SQLULEN size, SQLSMALLINT precision)
	{
		return {name, SQL_DECIMAL, size, precision, supports_null_values};
	}

	cpp_odbc::column_description make_numeric_column_description(SQLULEN size, SQLSMALLINT precision)
	{
		return {name, SQL_NUMERIC, size, precision, supports_null_values};
	}

}

void make_description_of_description_test::decimal_as_integer()
{
	test_as_integer(make_decimal_column_description(18, 0));
	test_as_integer(make_decimal_column_description(9, 0));
	test_as_integer(make_decimal_column_description(1, 0));
	test_as_integer(make_numeric_column_description(18, 0));
	test_as_integer(make_numeric_column_description(9, 0));
	test_as_integer(make_numeric_column_description(1, 0));
}

void make_description_of_description_test::decimal_as_floating_point()
{
	test_as_floating_point(make_decimal_column_description(18, 1));
	test_as_floating_point(make_numeric_column_description(18, 1));
}

void make_description_of_description_test::decimal_as_string()
{
	std::size_t const size = 19;
	// add three bytes to size (null-termination, sign, decimal point
	test_as_string(make_decimal_column_description(size, 0), size + 3);
	test_as_string(make_decimal_column_description(size, 5), size + 3);
	test_as_string(make_numeric_column_description(size, 0), size + 3);
	test_as_string(make_numeric_column_description(size, 5), size + 3);
}

/**
 *  @file make_description_test.cpp
 *  @date 13.02.2015
 *  @author mkoenig
 */

#include <pydbc/make_description.h>

#include <cppunit/extensions/HelperMacros.h>

#include <pydbc/description_types.h>

#include <sqlext.h>
#include <sstream>
#include <stdexcept>


class make_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( make_description_test );

	CPPUNIT_TEST( unsupported_type_throws );
	CPPUNIT_TEST( integer_types );
	CPPUNIT_TEST( string_types );
	CPPUNIT_TEST( bit_type );
	CPPUNIT_TEST( decimal_as_integer );

CPPUNIT_TEST_SUITE_END();

public:

	void unsupported_type_throws();
	void integer_types();
	void string_types();
	void bit_type();
	void decimal_as_integer();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( make_description_test );

using pydbc::make_description;


void make_description_test::unsupported_type_throws()
{
	SQLSMALLINT const unsupported_type = SQL_GUID;
	cpp_odbc::column_description column_description = {"dummy", unsupported_type, 0, 0, false};
	CPPUNIT_ASSERT_THROW( make_description(column_description), std::runtime_error );
}

void make_description_test::integer_types()
{
	std::vector<SQLSMALLINT> const types = {
			SQL_SMALLINT, SQL_INTEGER, SQL_TINYINT, SQL_BIGINT,
		};

	for (auto const type : types) {
		cpp_odbc::column_description column_description = {"dummy", type, 0, 0, false};
		auto const description = make_description(column_description);
		std::ostringstream message;
		message << "Could not convert type identifier '" << type << "' to integer description";
		CPPUNIT_ASSERT_MESSAGE( message.str(), dynamic_cast<pydbc::integer_description const *>(description.get()) );
	}
}

void make_description_test::string_types()
{
	std::vector<SQLSMALLINT> const types = {
			SQL_CHAR, SQL_VARCHAR, SQL_LONGVARCHAR, SQL_WCHAR, SQL_WVARCHAR, SQL_WLONGVARCHAR
		};

	SQLULEN const expected_size = 42;

	for (auto const type : types) {
		cpp_odbc::column_description column_description = {"dummy", type, expected_size - 1, 0, false};
		auto const description = make_description(column_description);
		std::ostringstream message;
		message << "Could not convert type identifier '" << type << "' to string description";
		CPPUNIT_ASSERT_MESSAGE( message.str(), dynamic_cast<pydbc::string_description const *>(description.get()) );
		CPPUNIT_ASSERT_EQUAL( expected_size, description->element_size() );
	}
}

void make_description_test::bit_type()
{
	SQLSMALLINT const type = SQL_BIT;

	cpp_odbc::column_description column_description = {"dummy", type, 0, 0, false};
	auto const description = make_description(column_description);
	CPPUNIT_ASSERT( dynamic_cast<pydbc::boolean_description const *>(description.get()) );
}

void make_description_test::decimal_as_integer()
{
	SQLSMALLINT const type = SQL_DECIMAL;

	cpp_odbc::column_description column_description = {"dummy", type, 0, 0, false};
	auto const description = make_description(column_description);
	CPPUNIT_ASSERT( dynamic_cast<pydbc::integer_description const *>(description.get()) );
}

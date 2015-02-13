/**
 *  @file description_test.cpp
 *  @date 06.02.2015
 *  @author mkoenig
 */

#include "pydbc/description_types.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class description_types_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( description_types_test );

	CPPUNIT_TEST( integer );
	CPPUNIT_TEST( string );
	CPPUNIT_TEST( boolean );
	CPPUNIT_TEST( floating_point );

CPPUNIT_TEST_SUITE_END();

public:

	void integer();
	void string();
	void boolean();
	void floating_point();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( description_types_test );

void description_types_test::integer()
{
	long const expected = 42;
	pydbc::integer_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(expected), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_SBIGINT, description.column_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

void description_types_test::string()
{
	std::size_t const size = 42;
	std::string const expected("test string");
	pydbc::string_description const description(size);

	CPPUNIT_ASSERT_EQUAL(size + 1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_CHAR, description.column_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(expected.c_str()));
}

void description_types_test::boolean()
{
	pydbc::boolean_description const description;

	CPPUNIT_ASSERT_EQUAL(1, description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_BIT, description.column_type());

	char as_bool = 0;
	CPPUNIT_ASSERT_EQUAL(pydbc::field{false}, description.make_field(&as_bool));
	as_bool = 1;
	CPPUNIT_ASSERT_EQUAL(pydbc::field{true}, description.make_field(&as_bool));
}

void description_types_test::floating_point()
{
	double const expected = 3.14;
	pydbc::floating_point_description const description;

	CPPUNIT_ASSERT_EQUAL(sizeof(expected), description.element_size());
	CPPUNIT_ASSERT_EQUAL(SQL_C_DOUBLE, description.column_type());
	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&expected)));
}

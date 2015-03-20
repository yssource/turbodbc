#include "pydbc/descriptions/number_description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <sqlext.h>


class number_description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( number_description_test );

	CPPUNIT_TEST( positive_integer );
	CPPUNIT_TEST( negative_integer );

CPPUNIT_TEST_SUITE_END();

public:

	void positive_integer();
	void negative_integer();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( number_description_test );

void number_description_test::positive_integer()
{
	SQL_NUMERIC_STRUCT data = {
				18,
				0,
				1, // + sign
				// binary representation of 1234567890 from low to high bytes
				{0xd2, 0x02, 0x96, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
			};

	long const expected = 1234567890;
	pydbc::number_description description;
	auto const actual = description.make_field(reinterpret_cast<char const *>(&data));

	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, actual);
}

void number_description_test::negative_integer()
{
	SQL_NUMERIC_STRUCT data = {
				18,
				0,
				0, // - sign
				// binary representation of 1234567890 from low to high bytes
				{0xd2, 0x02, 0x96, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
			};

	long const expected = -1234567890;
	pydbc::number_description description;
	auto const actual = description.make_field(reinterpret_cast<char const *>(&data));

	CPPUNIT_ASSERT_EQUAL(pydbc::field{expected}, actual);
}

/**
 *  @file fixed_length_string_buffer_test.cpp
 *  @date 07.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:57:29 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21205 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/level2/fixed_length_string_buffer.h"

#include <cstring>

class fixed_length_string_buffer_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( fixed_length_string_buffer_test );

	CPPUNIT_TEST( capacity );
	CPPUNIT_TEST( string_cast );

CPPUNIT_TEST_SUITE_END();

public:

	void capacity();
	void string_cast();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( fixed_length_string_buffer_test );

using cpp_odbc::level2::fixed_length_string_buffer;

void fixed_length_string_buffer_test::capacity()
{
	std::size_t const expected = 5;
	fixed_length_string_buffer<expected> const buffer;
	CPPUNIT_ASSERT_EQUAL( expected, buffer.capacity() );
}

void fixed_length_string_buffer_test::string_cast()
{
	fixed_length_string_buffer<5> buffer;

	std::string const expected = "dummy";
	memcpy(buffer.data_pointer(), expected.c_str(), expected.size());

	std::string const actual(buffer);
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

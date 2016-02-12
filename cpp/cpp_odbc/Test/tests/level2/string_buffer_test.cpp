/**
 *  @file string_buffer_test.cpp
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

#include "cpp_odbc/level2/string_buffer.h"

#include <cstring>

class string_buffer_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( string_buffer_test );

	CPPUNIT_TEST( capacity );
	CPPUNIT_TEST( string_cast );

CPPUNIT_TEST_SUITE_END();

public:

	void capacity();
	void string_cast();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( string_buffer_test );

void string_buffer_test::capacity()
{
	signed short int const expected_capacity = 1000;
	cpp_odbc::level2::string_buffer buffer(expected_capacity);

	CPPUNIT_ASSERT_EQUAL( expected_capacity, buffer.capacity() );
}

void string_buffer_test::string_cast()
{
	cpp_odbc::level2::string_buffer buffer(1000);

	std::string const expected = "test message";

	memcpy(buffer.data_pointer(), expected.c_str(), expected.size());
	*buffer.size_pointer() = expected.size();

	std::string const actual(buffer);

	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

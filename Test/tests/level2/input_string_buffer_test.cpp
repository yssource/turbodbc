/**
 *  @file input_string_buffer_test.cpp
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:57:29 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21205 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/extensions/assert_equal_with_different_types.h"

#include "cpp_odbc/level2/input_string_buffer.h"

class input_string_buffer_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( input_string_buffer_test );

	CPPUNIT_TEST( copies_value );
	CPPUNIT_TEST( has_trailing_slask_zero );

CPPUNIT_TEST_SUITE_END();

public:

	void copies_value();
	void has_trailing_slask_zero();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( input_string_buffer_test );

void input_string_buffer_test::copies_value()
{
	std::string const data("dummy data");
	cpp_odbc::level2::input_string_buffer buffer(data);

	CPPUNIT_ASSERT_EQUAL(data.size(), buffer.size());

	for (std::size_t i = 0; i < data.size(); ++i) {
		CPPUNIT_ASSERT_EQUAL( data[i], buffer.data_pointer()[i] );
	}
}

void input_string_buffer_test::has_trailing_slask_zero()
{
	std::string const data("dummy data");
	cpp_odbc::level2::input_string_buffer buffer(data);

	CPPUNIT_ASSERT_EQUAL(static_cast<unsigned char>('\0'), *(buffer.data_pointer() + buffer.size()));
}

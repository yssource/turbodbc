/**
 *  @file error_test.cpp
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
#include "cppunit_toolbox/extensions/assert_equal_with_different_types.h"

#include "cpp_odbc/error.h"
#include "cpp_odbc/level2/diagnostic_record.h"

class error_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( error_test );

	CPPUNIT_TEST( message_construction );
	CPPUNIT_TEST( diagnostic_construction );

CPPUNIT_TEST_SUITE_END();

public:

	void message_construction();
	void diagnostic_construction();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( error_test );

void error_test::message_construction()
{
	std::string const expected_message = "test message";
	cpp_odbc::error exception(expected_message);
	CPPUNIT_ASSERT_EQUAL(expected_message, exception.what());
}

void error_test::diagnostic_construction()
{
	cpp_odbc::level2::diagnostic_record const record = {"ABCDE", 42, "test message"};

	std::string const expected_message = "ODBC error\nstate: ABCDE\nnative error code: 42\nmessage: test message";

	cpp_odbc::error exception(record);
	CPPUNIT_ASSERT_EQUAL(expected_message, exception.what());
}

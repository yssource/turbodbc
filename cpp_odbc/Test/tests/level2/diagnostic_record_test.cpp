/**
 *  @file diagnostic_record_test.cpp
 *  @date 03.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:57:29 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21205 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/level2/diagnostic_record.h"

class diagnostic_record_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( diagnostic_record_test );

	CPPUNIT_TEST( test_members );

CPPUNIT_TEST_SUITE_END();

public:

	void test_members();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( diagnostic_record_test );

void diagnostic_record_test::test_members()
{
	std::string const odbc_state = "ABCDE";
	int const native_state = -1;
	std::string const message = "Everything is bad.";

	cpp_odbc::level2::diagnostic_record record = {odbc_state, native_state, message};

	CPPUNIT_ASSERT_EQUAL( odbc_state, record.odbc_status_code );
	CPPUNIT_ASSERT_EQUAL( native_state, record.native_error_code );
	CPPUNIT_ASSERT_EQUAL( message, record.message );
}

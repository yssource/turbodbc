/**
 *  @file credentials_test.cpp
 *  @date Sep 3, 2013
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-12-05 08:55:14 +0100 (Fr, 05 Dez 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21240 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/credentials.h"

class credentials_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( credentials_test );

	CPPUNIT_TEST( members );

CPPUNIT_TEST_SUITE_END();

public:

	void members()
	{
		std::string const user("username");
		std::string const pw("secret password");
		cpp_odbc::credentials creds = {user, pw};

		CPPUNIT_ASSERT_EQUAL(user, creds.user);
		CPPUNIT_ASSERT_EQUAL(pw, creds.password);
	}

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( credentials_test );

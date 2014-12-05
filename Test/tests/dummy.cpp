/**
 *  @file dummy.cpp
 *  @date 05.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */


#include <cppunit/extensions/HelperMacros.h>

class dummy : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( dummy );

	CPPUNIT_TEST( test );

CPPUNIT_TEST_SUITE_END();

public:

	void test();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( dummy );

void dummy::test()
{
	double expected = 1.0;
	double actual = 2.0;
	CPPUNIT_ASSERT_EQUAL_MESSAGE("Message", expected, actual);
}

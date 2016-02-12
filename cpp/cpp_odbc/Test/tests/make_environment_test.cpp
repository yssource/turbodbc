/**
 *  @file make_environment_test.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "cpp_odbc/make_environment.h"

#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/level3/raii_environment.h"
#include <memory>

class make_environment_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( make_environment_test );

	CPPUNIT_TEST( test_production_environment );
	CPPUNIT_TEST( test_debug_environment );

CPPUNIT_TEST_SUITE_END();

public:

	void test_production_environment()
	{
		auto environment = cpp_odbc::make_environment();
		bool const is_raii_environment =
				(std::dynamic_pointer_cast<cpp_odbc::level3::raii_environment>(environment) != nullptr);
		CPPUNIT_ASSERT( is_raii_environment );
	}

	void test_debug_environment()
	{
		auto environment = cpp_odbc::make_debug_environment();
		bool const is_raii_environment =
				(std::dynamic_pointer_cast<cpp_odbc::level3::raii_environment>(environment) != nullptr);
		CPPUNIT_ASSERT( is_raii_environment );
	}
};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( make_environment_test );

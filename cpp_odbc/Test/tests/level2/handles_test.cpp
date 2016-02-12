/**
 *  @file handles_test.cpp
 *  @date 13.03.2014
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

#include "cpp_odbc/level2/handles.h"

#include "sql.h"

class handles_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( handles_test );

	CPPUNIT_TEST( connection_handle );
	CPPUNIT_TEST( connection_handle_equality );
	CPPUNIT_TEST( environment_handle );
	CPPUNIT_TEST( environment_handle_equality );
	CPPUNIT_TEST( statement_handle );
	CPPUNIT_TEST( statement_handle_equality );

CPPUNIT_TEST_SUITE_END();

public:

	void connection_handle();
	void connection_handle_equality();
	void environment_handle();
	void environment_handle_equality();
	void statement_handle();
	void statement_handle_equality();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( handles_test );

namespace {
	//destinations for pointers
	int const dummy_value = 23;
	int value_a = dummy_value;
	int value_b = dummy_value;

	template <typename Handle>
	void test_handle(signed short int expected_type)
	{
		Handle const handle = {&value_a};
		CPPUNIT_ASSERT( &value_a == handle.handle );
		CPPUNIT_ASSERT_EQUAL( expected_type, handle.type() );
	}

	template <typename Handle>
	void test_handle_equality()
	{
		Handle const handle_a = {&value_a};
		Handle const handle_b = {&value_b};

		CPPUNIT_ASSERT( handle_a == handle_a );
		CPPUNIT_ASSERT( not (handle_a == handle_b) );
		CPPUNIT_ASSERT( handle_a != handle_b );
		CPPUNIT_ASSERT( not (handle_a != handle_a) );
	}

}


void handles_test::connection_handle()
{
	test_handle<cpp_odbc::level2::connection_handle>(SQL_HANDLE_DBC);
}

void handles_test::connection_handle_equality()
{
	test_handle_equality<cpp_odbc::level2::connection_handle>();
}

void handles_test::environment_handle()
{
	test_handle<cpp_odbc::level2::environment_handle>(SQL_HANDLE_ENV);
}

void handles_test::environment_handle_equality()
{
	test_handle_equality<cpp_odbc::level2::environment_handle>();
}

void handles_test::statement_handle()
{
	test_handle<cpp_odbc::level2::statement_handle>(SQL_HANDLE_STMT);
}

void handles_test::statement_handle_equality()
{
	test_handle_equality<cpp_odbc::level2::statement_handle>();
}

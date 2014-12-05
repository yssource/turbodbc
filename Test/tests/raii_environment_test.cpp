/**
 *  @file raii_environment_test.cpp
 *  @date 13.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/raii_environment.h"

#include "cpp_odbc_test/level2_mock_api.h"
#include "cpp_odbc_test/level2_dummy_api.h"

#include "sqlext.h"

#include <type_traits>

class raii_environment_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( raii_environment_test );

	CPPUNIT_TEST( sets_odbc_version );
	CPPUNIT_TEST( get_api );
	CPPUNIT_TEST( get_handle );

CPPUNIT_TEST_SUITE_END();

public:

	void sets_odbc_version();
	void get_api();
	void get_handle();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( raii_environment_test );

namespace {
	// used as destination for pointers
	int value = 42;
}

namespace level2 = cpp_odbc::level2;
using cpp_odbc::raii_environment;

void raii_environment_test::sets_odbc_version()
{
	level2::environment_handle internal_handle = {&value};

	auto api = psapp::make_valid_ptr<cpp_odbc_test::level2_mock_api const>();

	EXPECT_CALL(*api, do_allocate_environment_handle()).
			WillOnce(testing::Return(internal_handle));
	EXPECT_CALL(*api, do_set_environment_attribute(internal_handle, SQL_ATTR_ODBC_VERSION, SQL_OV_ODBC3)).
			Times(1);

	// check that free_handle is called on destruction
	{
		raii_environment instance(api);

		EXPECT_CALL(*api, do_free_handle(internal_handle)).
				Times(1);
	}
}

void raii_environment_test::get_api()
{
	auto expected_api = psapp::make_valid_ptr<cpp_odbc_test::level2_dummy_api const>();

	raii_environment instance(expected_api);
	CPPUNIT_ASSERT( expected_api == instance.get_api());
}

void raii_environment_test::get_handle()
{
	auto expected_api = psapp::make_valid_ptr<cpp_odbc_test::level2_dummy_api const>();

	raii_environment instance(expected_api);
	bool const returns_handle_ref = std::is_same<level2::environment_handle const &, decltype(instance.get_handle())>::value;

	CPPUNIT_ASSERT( returns_handle_ref );
	instance.get_handle(); // make sure function symbol is there
}

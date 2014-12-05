/**
 *  @file raii_connection_test.cpp
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */


#include "cpp_odbc/raii_connection.h"

#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc_test/level2_mock_api.h"

class raii_connection_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( raii_connection_test );

	CPPUNIT_TEST( raii_connect_and_disconnect );

CPPUNIT_TEST_SUITE_END();

public:

	void raii_connect_and_disconnect();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( raii_connection_test );

using cpp_odbc::raii_connection;
using cpp_odbc_test::level2_mock_api;
using cpp_odbc::level2::environment_handle;
using cpp_odbc::level2::connection_handle;

namespace {

	// destinations for pointers, values irrelevant
	int value_a = 17;
	int value_b = 23;

	environment_handle const e_handle = {&value_a};
	connection_handle const default_c_handle = {&value_b};

	psapp::valid_ptr<testing::NiceMock<level2_mock_api>> make_default_api()
	{
		auto api = psapp::make_valid_ptr<testing::NiceMock<level2_mock_api>>();

		ON_CALL(*api, do_allocate_connection_handle(testing::_))
			.WillByDefault(testing::Return(default_c_handle));

		return api;
	}

}

void raii_connection_test::raii_connect_and_disconnect()
{
	auto api = psapp::make_valid_ptr<cpp_odbc_test::level2_mock_api>();
	connection_handle c_handle = {&value_a};

	std::string const connection_string = "my DSN";

	EXPECT_CALL(*api, do_allocate_connection_handle(e_handle)).
			WillOnce(testing::Return(c_handle));
	EXPECT_CALL(*api, do_establish_connection(c_handle, connection_string)).Times(1);

	{ // scope introduced for RAII check
		raii_connection connection(api, e_handle, connection_string);
		EXPECT_CALL(*api, do_disconnect(c_handle)).Times(1);
		EXPECT_CALL(*api, do_free_handle(c_handle)).Times(1);
	}
}


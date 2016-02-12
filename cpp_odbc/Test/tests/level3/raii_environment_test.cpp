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

#include "cpp_odbc/level3/raii_environment.h"

#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/level3/raii_connection.h"
#include "cpp_odbc_test/level2_mock_api.h"

#include "sqlext.h"

#include <type_traits>

class raii_environment_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( raii_environment_test );

	CPPUNIT_TEST( is_environment );
	CPPUNIT_TEST( resource_management );
	CPPUNIT_TEST( sets_odbc_version_on_construction );
	CPPUNIT_TEST( get_api );
	CPPUNIT_TEST( get_handle );

	CPPUNIT_TEST( make_connection );
	CPPUNIT_TEST( set_attribute );

CPPUNIT_TEST_SUITE_END();

public:

	void is_environment();
	void resource_management();
	void sets_odbc_version_on_construction();
	void get_api();
	void get_handle();

	void make_connection();
	void set_attribute();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( raii_environment_test );

using cpp_odbc::level3::raii_connection;
using cpp_odbc::level3::raii_environment;
using cpp_odbc_test::level2_mock_api;
using cpp_odbc::level2::environment_handle;
using cpp_odbc::level2::connection_handle;

namespace {
	// destinations for pointers, values irrelevant
	int value_a = 17;
	int value_b = 23;

	environment_handle const default_e_handle = {&value_a};
	connection_handle const default_c_handle = {&value_b};

	std::shared_ptr<testing::NiceMock<level2_mock_api> const> make_default_api()
	{
		auto api = std::make_shared<testing::NiceMock<level2_mock_api> const>();

		ON_CALL(*api, do_allocate_environment_handle())
			.WillByDefault(testing::Return(default_e_handle));
		ON_CALL(*api, do_allocate_connection_handle(testing::_))
			.WillByDefault(testing::Return(default_c_handle));

		return api;
	}

}

void raii_environment_test::is_environment()
{
	bool const derived_from_environment = std::is_base_of<cpp_odbc::environment, raii_environment>::value;
	CPPUNIT_ASSERT( derived_from_environment );
}

void raii_environment_test::resource_management()
{
	environment_handle internal_handle = {&value_a};

	auto api = std::make_shared<testing::NiceMock<cpp_odbc_test::level2_mock_api> const>();

	EXPECT_CALL(*api, do_allocate_environment_handle()).
			WillOnce(testing::Return(internal_handle));

	// check that free_handle is called on destruction
	{
		raii_environment instance(api);

		EXPECT_CALL(*api, do_free_handle(internal_handle)).
				Times(1);
	}
}

void raii_environment_test::sets_odbc_version_on_construction()
{
	auto api = make_default_api();
	EXPECT_CALL(*api, do_set_environment_attribute(default_e_handle, SQL_ATTR_ODBC_VERSION, SQL_OV_ODBC3)).
			Times(1);

	raii_environment instance(api);
}

void raii_environment_test::get_api()
{
	auto expected_api = make_default_api();

	raii_environment instance(expected_api);
	CPPUNIT_ASSERT( expected_api == instance.get_api());
}

void raii_environment_test::get_handle()
{
	auto api = make_default_api();

	raii_environment instance(api);
	bool const returns_handle_ref = std::is_same<environment_handle const &, decltype(instance.get_handle())>::value;

	CPPUNIT_ASSERT( returns_handle_ref );
	instance.get_handle(); // make sure function symbol is there
}

void raii_environment_test::make_connection()
{
	std::string const connection_string("dummy connection string");
	auto api = make_default_api();

	auto environment = std::make_shared<raii_environment const>(api);
	EXPECT_CALL(*api, do_allocate_connection_handle(default_e_handle))
		.WillOnce(testing::Return(default_c_handle));
	EXPECT_CALL(*api, do_establish_connection(testing::_, connection_string))
		.Times(1);

	auto connection = environment->make_connection(connection_string);
	bool const is_raii_connection = (std::dynamic_pointer_cast<raii_connection const>(connection) != nullptr);
	CPPUNIT_ASSERT( is_raii_connection );
}

void raii_environment_test::set_attribute()
{
	SQLINTEGER const attribute = 42;
	long const value = 1234;

	auto api = make_default_api();
	raii_environment environment(api);

	EXPECT_CALL(*api, do_set_environment_attribute(default_e_handle, attribute, value))
		.Times(1);

	environment.set_attribute(attribute, value);
}

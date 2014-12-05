/**
 *  @file environment_test.cpp
 *  @date 16.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc/environment.h"

#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/helpers/is_abstract_base_class.h"

#include "cpp_odbc_test/mock_connection.h"

class environment_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( environment_test );

	CPPUNIT_TEST( is_suitable_as_base_class );
	CPPUNIT_TEST( make_connection_forwards );
	CPPUNIT_TEST( set_integer_attribute_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void is_suitable_as_base_class();
	void make_connection_forwards();
	void set_integer_attribute_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( environment_test );

namespace {

	class mock_environment : public cpp_odbc::environment {
	public:
		MOCK_CONST_METHOD1(do_make_connection, std::shared_ptr<cpp_odbc::connection>(std::string const &));
		MOCK_CONST_METHOD2(do_set_attribute, void(SQLINTEGER, long));
	};

}

void environment_test::is_suitable_as_base_class()
{
	bool const is_base = cppunit_toolbox::is_abstract_base_class<cpp_odbc::environment>::value;
	CPPUNIT_ASSERT( is_base );
}

void environment_test::make_connection_forwards()
{
	mock_environment environment;
	std::string const connection_string("test DSN");
	auto expected = std::make_shared<cpp_odbc_test::mock_connection>();

	EXPECT_CALL(environment, do_make_connection(connection_string))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT( expected == environment.make_connection(connection_string) );
}

void environment_test::set_integer_attribute_forwards()
{
	mock_environment environment;
	SQLINTEGER const attribute = 42;
	long const value = 17;

	EXPECT_CALL(environment, do_set_attribute(attribute, value)).Times(1);

	environment.set_attribute(attribute, value);
}

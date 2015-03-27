/**
 *  @file description_test.cpp
 *  @date 05.02.2015
 *  @author mkoenig
 */

#include "pydbc/description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/helpers/is_abstract_base_class.h>

#include <gmock/gmock.h>


class description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( description_test );

	CPPUNIT_TEST( is_base_class );
	CPPUNIT_TEST( element_size_forwards );
	CPPUNIT_TEST( column_type_forwards );
	CPPUNIT_TEST( column_sql_type_forwards );
	CPPUNIT_TEST( make_field_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void is_base_class();
	void element_size_forwards();
	void column_type_forwards();
	void column_sql_type_forwards();
	void make_field_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( description_test );

namespace {

	struct mock_description : public pydbc::description {
		MOCK_CONST_METHOD0(do_element_size, std::size_t());
		MOCK_CONST_METHOD0(do_column_c_type, SQLSMALLINT());
		MOCK_CONST_METHOD0(do_column_sql_type, SQLSMALLINT());
		MOCK_CONST_METHOD1(do_make_field, pydbc::field(char const *));
	};

}

void description_test::is_base_class()
{
	bool const all_good = cppunit_toolbox::is_abstract_base_class<pydbc::description>::value;
	CPPUNIT_ASSERT(all_good);
}

void description_test::element_size_forwards()
{
	std::size_t const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_element_size())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.element_size());
}

void description_test::column_type_forwards()
{
	SQLSMALLINT const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_column_c_type())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.column_c_type());
}

void description_test::column_sql_type_forwards()
{
	SQLSMALLINT const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_column_sql_type())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.column_sql_type());
}

void description_test::make_field_forwards()
{
	pydbc::field const expected(42l);
	char const * data = nullptr;

	mock_description description;
	EXPECT_CALL(description, do_make_field(data))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.make_field(data));
}

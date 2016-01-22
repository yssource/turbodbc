#include "turbodbc/description.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>
#include <cppunit_toolbox/helpers/is_abstract_base_class.h>

#include <gmock/gmock.h>


class description_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( description_test );

	CPPUNIT_TEST( is_base_class );
	CPPUNIT_TEST( element_size_forwards );
	CPPUNIT_TEST( column_type_forwards );
	CPPUNIT_TEST( column_sql_type_forwards );
	CPPUNIT_TEST( make_field_forwards );
	CPPUNIT_TEST( set_field_forwards );
	CPPUNIT_TEST( type_code_forwards );
	CPPUNIT_TEST( default_name );
	CPPUNIT_TEST( default_supports_null_values );
	CPPUNIT_TEST( custom_name_and_nullable_support );

CPPUNIT_TEST_SUITE_END();

public:

	void is_base_class();
	void element_size_forwards();
	void column_type_forwards();
	void column_sql_type_forwards();
	void make_field_forwards();
	void set_field_forwards();
	void type_code_forwards();
	void default_name();
	void default_supports_null_values();
	void custom_name_and_nullable_support();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( description_test );

namespace {

	struct mock_description : public turbodbc::description {
		mock_description() : turbodbc::description() {}
		mock_description(std::string name, bool supports_null_values) :
			turbodbc::description(std::move(name), supports_null_values)
		{
		}

		MOCK_CONST_METHOD0(do_element_size, std::size_t());
		MOCK_CONST_METHOD0(do_column_c_type, SQLSMALLINT());
		MOCK_CONST_METHOD0(do_column_sql_type, SQLSMALLINT());
		MOCK_CONST_METHOD1(do_make_field, turbodbc::field(char const *));
		MOCK_CONST_METHOD2(do_set_field, void(cpp_odbc::writable_buffer_element &, turbodbc::field const &));
		MOCK_CONST_METHOD0(do_get_type_code, turbodbc::type_code());
	};

}

void description_test::is_base_class()
{
	bool const all_good = cppunit_toolbox::is_abstract_base_class<turbodbc::description>::value;
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
	turbodbc::field const expected(42l);
	char const * data = nullptr;

	mock_description description;
	EXPECT_CALL(description, do_make_field(data))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.make_field(data));
}

void description_test::set_field_forwards()
{
	turbodbc::field const value(42l);
	cpp_odbc::multi_value_buffer buffer(42, 10);
	auto element = buffer[0];

	mock_description description;
	EXPECT_CALL(description, do_set_field(testing::Ref(element), value)).Times(1);

	CPPUNIT_ASSERT_NO_THROW(description.set_field(element, value));
}

void description_test::type_code_forwards()
{
	auto const expected = turbodbc::type_code::string;
	mock_description description;
	EXPECT_CALL(description, do_get_type_code())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == description.get_type_code());
}

void description_test::default_name()
{
	mock_description description;

	CPPUNIT_ASSERT_EQUAL("parameter", description.name());
}

void description_test::default_supports_null_values()
{
	mock_description description;

	CPPUNIT_ASSERT(description.supports_null_values());
}

void description_test::custom_name_and_nullable_support()
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;
	mock_description description(expected_name, expected_supports_null);

	CPPUNIT_ASSERT_EQUAL(expected_name, description.name());
	CPPUNIT_ASSERT_EQUAL(expected_supports_null, description.supports_null_values());
}

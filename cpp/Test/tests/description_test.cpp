#include "turbodbc/description.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>


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


TEST(DescriptionTest, ElementSizeForwards)
{
	std::size_t const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_element_size())
		.WillOnce(testing::Return(expected));

	EXPECT_EQ(expected, description.element_size());
}

TEST(DescriptionTest, ColumnTypeForwards)
{
	SQLSMALLINT const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_column_c_type())
		.WillOnce(testing::Return(expected));

	EXPECT_EQ(expected, description.column_c_type());
}

TEST(DescriptionTest, ColumnSqlTypeForwards)
{
	SQLSMALLINT const expected = 42;

	mock_description description;
	EXPECT_CALL(description, do_column_sql_type())
		.WillOnce(testing::Return(expected));

	EXPECT_EQ(expected, description.column_sql_type());
}

TEST(DescriptionTest, MakeFieldForwards)
{
	turbodbc::field const expected(42l);
	char const * data = nullptr;

	mock_description description;
	EXPECT_CALL(description, do_make_field(data))
		.WillOnce(testing::Return(expected));

	EXPECT_EQ(expected, description.make_field(data));
}

TEST(DescriptionTest, SetFieldForwards)
{
	turbodbc::field const value(42l);
	cpp_odbc::multi_value_buffer buffer(42, 10);
	auto element = buffer[0];

	mock_description description;
	EXPECT_CALL(description, do_set_field(testing::Ref(element), value)).Times(1);

	ASSERT_NO_THROW(description.set_field(element, value));
}

TEST(DescriptionTest, TypeCodeForwards)
{
	auto const expected = turbodbc::type_code::string;
	mock_description description;
	EXPECT_CALL(description, do_get_type_code())
		.WillOnce(testing::Return(expected));

	EXPECT_EQ(expected, description.get_type_code());
}

TEST(DescriptionTest, DefaultName)
{
	mock_description description;

	EXPECT_EQ("parameter", description.name());
}

TEST(DescriptionTest, DefaultSupportsNullValues)
{
	mock_description description;

	EXPECT_TRUE(description.supports_null_values());
}

TEST(DescriptionTest, CustomNameAndNullableSupport)
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;
	mock_description description(expected_name, expected_supports_null);

	EXPECT_EQ(expected_name, description.name());
	EXPECT_EQ(expected_supports_null, description.supports_null_values());
}

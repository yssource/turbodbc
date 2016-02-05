#include "turbodbc/descriptions/boolean_description.h"

#include <gtest/gtest.h>
#include <sqlext.h>

TEST(BooleanDescriptionTest, BasicProperties)
{
	turbodbc::boolean_description const description;
	EXPECT_EQ(1, description.element_size());
	EXPECT_EQ(SQL_C_BIT, description.column_c_type());
	EXPECT_EQ(SQL_BIT, description.column_sql_type());
}

TEST(BooleanDescriptionTest, MakeField)
{
	turbodbc::boolean_description const description;

	char as_bool = 0;
	EXPECT_EQ(turbodbc::field{false}, description.make_field(&as_bool));
	as_bool = 1;
	EXPECT_EQ(turbodbc::field{true}, description.make_field(&as_bool));
}

TEST(BooleanDescriptionTest, SetField)
{
	turbodbc::boolean_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{true});
	EXPECT_EQ(1, *element.data_pointer);
	EXPECT_EQ(description.element_size(), element.indicator);

	description.set_field(element, turbodbc::field{false});
	EXPECT_EQ(0, *element.data_pointer);
	EXPECT_EQ(description.element_size(), element.indicator);
}

TEST(BooleanDescriptionTest, GetTypeCode)
{
	turbodbc::boolean_description const description;
	EXPECT_EQ( turbodbc::type_code::boolean, description.get_type_code() );
}

TEST(BooleanDescriptionTest, CustomNameAndNullableSupport)
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::boolean_description const description(expected_name, expected_supports_null);

	EXPECT_EQ(expected_name, description.name());
	EXPECT_EQ(expected_supports_null, description.supports_null_values());
}

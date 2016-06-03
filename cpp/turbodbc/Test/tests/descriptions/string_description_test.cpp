#include "turbodbc/descriptions/string_description.h"

#include <gtest/gtest.h>
#include <sqlext.h>


TEST(StringDescriptionTest, BasicProperties)
{
	std::size_t const size = 42;
	turbodbc::string_description const description(size);

	EXPECT_EQ(size + 1, description.element_size());
	EXPECT_EQ(SQL_C_CHAR, description.column_c_type());
	EXPECT_EQ(SQL_VARCHAR, description.column_sql_type());
}

TEST(StringDescriptionTest, SetField)
{
	std::string const expected("another test string");
	turbodbc::string_description const description(expected.size() + 1); // one larger

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{expected});
	EXPECT_EQ(expected, std::string(element.data_pointer));
	EXPECT_EQ(expected.size(), element.indicator);
}

TEST(StringDescriptionTest, SetFieldWithMaximumLength)
{
	std::string const expected("another test string");
	turbodbc::string_description const description(expected.size());

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{expected});
	EXPECT_EQ(expected, std::string(element.data_pointer));
	EXPECT_EQ(expected.size(), element.indicator);
}

TEST(StringDescriptionTest, SetFieldThrowsForTooLongValues)
{
	std::string const basic("another test string");
	std::string const full(basic + "x");
	turbodbc::string_description const description(basic.size());

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	ASSERT_THROW(description.set_field(element, turbodbc::field{full}), std::runtime_error);
}

TEST(StringDescriptionTest, GetTypeCode)
{
	turbodbc::string_description const description(10);
	EXPECT_EQ(turbodbc::type_code::string, description.get_type_code());
}

TEST(StringDescriptionTest, CustomNameAndNullableSupport)
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::string_description const description(expected_name, expected_supports_null, 10);

	EXPECT_EQ(expected_name, description.name());
	EXPECT_EQ(expected_supports_null, description.supports_null_values());
}

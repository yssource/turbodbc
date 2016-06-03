#include "turbodbc/descriptions/timestamp_description.h"

#include <gtest/gtest.h>
#include <sqlext.h>


TEST(TimestampDescriptionTest, BasicProperties)
{
	turbodbc::timestamp_description const description;

	EXPECT_EQ(16, description.element_size());
	EXPECT_EQ(SQL_C_TYPE_TIMESTAMP, description.column_c_type());
	EXPECT_EQ(SQL_TYPE_TIMESTAMP, description.column_sql_type());
}

TEST(TimestampDescriptionTest, SetField)
{
	boost::posix_time::ptime const timestamp{{2015, 12, 31}, {1, 2, 3, 123456}};
	turbodbc::timestamp_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{timestamp});
	auto const as_sql_date = reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(element.data_pointer);
	EXPECT_EQ(2015, as_sql_date->year);
	EXPECT_EQ(12, as_sql_date->month);
	EXPECT_EQ(31, as_sql_date->day);
	EXPECT_EQ(1, as_sql_date->hour);
	EXPECT_EQ(2, as_sql_date->minute);
	EXPECT_EQ(3, as_sql_date->second);
	EXPECT_EQ(123456000, as_sql_date->fraction);
	EXPECT_EQ(description.element_size(), element.indicator);
}

TEST(TimestampDescriptionTest, GetTypeCode)
{
	turbodbc::timestamp_description const description;
	EXPECT_EQ(turbodbc::type_code::timestamp, description.get_type_code());
}

TEST(TimestampDescriptionTest, CustomNameAndNullableSupport)
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::timestamp_description const description(expected_name, expected_supports_null);

	EXPECT_EQ(expected_name, description.name());
	EXPECT_EQ(expected_supports_null, description.supports_null_values());
}

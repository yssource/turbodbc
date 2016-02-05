#include "turbodbc/descriptions/date_description.h"

#include <gtest/gtest.h>
#include <sqlext.h>

TEST(DateDescriptionTest, BasicProperties)
{
	turbodbc::date_description const description;

	EXPECT_EQ(6, description.element_size());
	EXPECT_EQ(SQL_C_TYPE_DATE, description.column_c_type());
	EXPECT_EQ(SQL_TYPE_DATE, description.column_sql_type());
}


TEST(DateDescriptionTest, MakeField)
{
	boost::gregorian::date const expected{2015, 12, 31};
	turbodbc::date_description const description;

	SQL_DATE_STRUCT const sql_date = {2015, 12, 31};
	EXPECT_EQ(turbodbc::field{expected}, description.make_field(reinterpret_cast<char const *>(&sql_date)));
}

TEST(DateDescriptionTest, SetField)
{
	boost::gregorian::date const date{2015, 12, 31};
	turbodbc::date_description const description;

	cpp_odbc::multi_value_buffer buffer(description.element_size(), 1);
	auto element = buffer[0];

	description.set_field(element, turbodbc::field{date});
	auto const as_sql_date = reinterpret_cast<SQL_DATE_STRUCT const *>(element.data_pointer);
	EXPECT_EQ(2015, as_sql_date->year);
	EXPECT_EQ(12, as_sql_date->month);
	EXPECT_EQ(31, as_sql_date->day);
	EXPECT_EQ(description.element_size(), element.indicator);
}

TEST(DateDescriptionTest, GetTypeCode)
{
	turbodbc::date_description const description;
	EXPECT_EQ( turbodbc::type_code::date, description.get_type_code() );
}

TEST(DateDescriptionTest, CustomNameAndNullableSupport)
{
	std::string const expected_name("my_name");
	bool const expected_supports_null = false;

	turbodbc::date_description const description(expected_name, expected_supports_null);

	EXPECT_EQ(expected_name, description.name());
	EXPECT_EQ(expected_supports_null, description.supports_null_values());
}

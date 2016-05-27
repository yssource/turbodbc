#include "turbodbc/field_translators/timestamp_translator.h"

#include <gtest/gtest.h>


using turbodbc::field_translators::timestamp_translator;

TEST(TimestampTranslatorTest, MakeField)
{
	cpp_odbc::multi_value_buffer buffer(1, 1);
	auto element = buffer[0];
	element.indicator = 1;
	auto const & as_const = buffer;

	timestamp_translator const translator;

	*reinterpret_cast<SQL_TIMESTAMP_STRUCT *>(element.data_pointer) = {2015, 12, 31, 1, 2, 3, 123456000};
	boost::posix_time::ptime const expected{{2015, 12, 31}, {1, 2, 3, 123456}};
	EXPECT_EQ(turbodbc::field(expected), *(translator.make_field(as_const[0])));
}

TEST(TimestampTranslatorTest, SetField)
{
	boost::posix_time::ptime const timestamp{{2015, 12, 31}, {1, 2, 3, 123456}};
	timestamp_translator const translator;

	cpp_odbc::multi_value_buffer buffer(sizeof(SQL_DATE_STRUCT), 1);
	auto element = buffer[0];

	translator.set_field(element, {turbodbc::field{timestamp}});
	auto const as_sql_ts = reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(element.data_pointer);
	EXPECT_EQ(2015, as_sql_ts->year);
	EXPECT_EQ(12, as_sql_ts->month);
	EXPECT_EQ(31, as_sql_ts->day);
	EXPECT_EQ(1, as_sql_ts->hour);
	EXPECT_EQ(2, as_sql_ts->minute);
	EXPECT_EQ(3, as_sql_ts->second);
	EXPECT_EQ(123456000, as_sql_ts->fraction);
	EXPECT_EQ(sizeof(SQL_TIMESTAMP_STRUCT), element.indicator);
}

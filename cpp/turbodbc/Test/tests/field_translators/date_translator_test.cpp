#include "turbodbc/field_translators/date_translator.h"

#include <gtest/gtest.h>


using turbodbc::field_translators::date_translator;

TEST(DateTranslatorTest, MakeField)
{
	cpp_odbc::multi_value_buffer buffer(1, 1);
	auto element = buffer[0];
	element.indicator = 1;
	auto const & as_const = buffer;

	date_translator const translator;

	*reinterpret_cast<SQL_DATE_STRUCT *>(element.data_pointer) = {2015, 12, 31};
	boost::gregorian::date const expected{2015, 12, 31};
	EXPECT_EQ(turbodbc::field(expected), *(translator.make_field(as_const[0])));
}


TEST(DateTranslatorTest, SetField)
{
	boost::gregorian::date const date{2015, 12, 31};
	date_translator const translator;

	cpp_odbc::multi_value_buffer buffer(sizeof(SQL_DATE_STRUCT), 1);
	auto element = buffer[0];

	translator.set_field(element, {turbodbc::field{date}});
	auto const as_sql_date = reinterpret_cast<SQL_DATE_STRUCT const *>(element.data_pointer);
	EXPECT_EQ(2015, as_sql_date->year);
	EXPECT_EQ(12, as_sql_date->month);
	EXPECT_EQ(31, as_sql_date->day);
	EXPECT_EQ(sizeof(SQL_DATE_STRUCT), element.indicator);
}

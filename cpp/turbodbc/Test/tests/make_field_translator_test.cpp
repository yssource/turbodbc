#include <turbodbc/make_field_translator.h>
#include <turbodbc/field_translators.h>

#include <gtest/gtest.h>

#include <stdexcept>


using turbodbc::make_field_translator;


TEST(MakeFieldTranslatorTest, UnsupportedTypeThrows)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::integer, true};
	info.type = static_cast<turbodbc::type_code>(-666); // assign invalid code
	EXPECT_THROW(make_field_translator(info), std::logic_error);
}

TEST(MakeFieldTranslatorTest, BooleanType)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::boolean, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::boolean_translator const *>(make_field_translator(info).get()));
}

TEST(MakeFieldTranslatorTest, DateType)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::date, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::date_translator const *>(make_field_translator(info).get()));
}

TEST(MakeFieldTranslatorTest, Float64Type)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::floating_point, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::float64_translator const *>(make_field_translator(info).get()));
}

TEST(MakeFieldTranslatorTest, Int64Type)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::integer, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::int64_translator const *>(make_field_translator(info).get()));
}

TEST(MakeFieldTranslatorTest, StringType)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::string, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::string_translator const *>(make_field_translator(info).get()));
}

TEST(MakeFieldTranslatorTest, TimestampType)
{
	turbodbc::column_info info = {"name", turbodbc::type_code::timestamp, true};
	EXPECT_TRUE(dynamic_cast<turbodbc::field_translators::timestamp_translator const *>(make_field_translator(info).get()));
}

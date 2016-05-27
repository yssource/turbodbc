#include "turbodbc/field_translators/float64_translator.h"

#include <gtest/gtest.h>


using turbodbc::field_translators::float64_translator;

TEST(Float64TranslatorTest, MakeField)
{
	cpp_odbc::multi_value_buffer buffer(8, 1);
	auto element = buffer[0];
	element.indicator = 8;
	auto const & as_const = buffer;

	float64_translator const translator;

	*reinterpret_cast<double *>(element.data_pointer) = 3.14;
	EXPECT_EQ(turbodbc::field{3.14}, *(translator.make_field(as_const[0])));
}


TEST(Float64TranslatorTest, SetField)
{
	float64_translator const translator;

	cpp_odbc::multi_value_buffer buffer(8, 1);
	auto element = buffer[0];

	translator.set_field(element, {turbodbc::field{3.14}});
	EXPECT_EQ(3.14, *reinterpret_cast<double *>(element.data_pointer));
	EXPECT_EQ(8, element.indicator);
}

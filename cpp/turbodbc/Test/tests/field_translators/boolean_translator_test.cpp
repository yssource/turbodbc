#include "turbodbc/field_translators/boolean_translator.h"

#include <gtest/gtest.h>


using turbodbc::field_translators::boolean_translator;

TEST(BooleanTranslatorTest, MakeField)
{
	cpp_odbc::multi_value_buffer buffer(1, 1);
	auto element = buffer[0];
	element.indicator = 1;
	auto const & as_const = buffer;

	boolean_translator const translator;

	*element.data_pointer = 0;
	EXPECT_EQ(turbodbc::field{false}, *(translator.make_field(as_const[0])));
	*element.data_pointer = 1;
	EXPECT_EQ(turbodbc::field{true}, *(translator.make_field(as_const[0])));
}


TEST(BooleanTranslatorTest, SetField)
{
	boolean_translator const translator;

	cpp_odbc::multi_value_buffer buffer(1, 1);
	auto element = buffer[0];

	translator.set_field(element, {turbodbc::field{true}});
	EXPECT_EQ(1, *element.data_pointer);
	EXPECT_EQ(1, element.indicator);

	translator.set_field(element, {turbodbc::field{false}});
	EXPECT_EQ(0, *element.data_pointer);
	EXPECT_EQ(1, element.indicator);
}

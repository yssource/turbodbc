#include "turbodbc/parameter.h"

#include <gtest/gtest.h>
#include "mock_classes.h"

#include <turbodbc/descriptions/integer_description.h>
#include <turbodbc/descriptions/string_description.h>
#include <boost/variant/get.hpp>

namespace {

	void set_buffer_element_to_null(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index)
	{
		auto element = buffer[row_index];
		element.indicator = SQL_NULL_DATA;
	}

	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg3;
	}

	std::size_t const parameter_index = 42;

}

TEST(ParameterTest, SetNonNullable)
{
	std::unique_ptr<turbodbc::integer_description> description(new turbodbc::integer_description());

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	ASSERT_TRUE( buffer != nullptr);

	auto const row_index = 42;
	long const expected = 123;
	parameter.set(row_index, turbodbc::field{expected});

	auto const actual = *reinterpret_cast<long *>((*buffer)[row_index].data_pointer);
	EXPECT_EQ(expected, actual);
}

TEST(ParameterTest, SetNullable)
{
	std::unique_ptr<turbodbc::integer_description> description(new turbodbc::integer_description());

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	ASSERT_TRUE( buffer != nullptr);

	auto const row_index = 42;
	turbodbc::nullable_field null;
	parameter.set(row_index, null);

	EXPECT_EQ(SQL_NULL_DATA, (*buffer)[row_index].indicator);
}


TEST(ParameterTest, GetBuffer)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(10));

	turbodbc_test::mock_statement statement;

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));

	auto const row_index = 42;
	std::string const expected("hi there!");
	parameter.set(row_index, turbodbc::field{expected});

	auto const & buffer = parameter.get_buffer();

	std::string const actual_content(buffer[row_index].data_pointer);
	auto const actual_indicator(buffer[row_index].indicator);
	EXPECT_EQ(expected, actual_content);
	EXPECT_EQ(expected.size(), actual_indicator);
}


TEST(ParameterTest, MoveToTop)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(10));

	turbodbc_test::mock_statement statement;

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));

	auto const row_index = 42;
	std::string const expected("hi there!");
	parameter.set(row_index, turbodbc::field{expected});

	move_to_top(parameter, row_index);

	auto const & buffer = parameter.get_buffer();

	std::string const top_content(buffer[0].data_pointer);
	auto const top_indicator(buffer[0].indicator);
	EXPECT_EQ(expected, top_content);
	EXPECT_EQ(expected.size(), top_indicator);
}


TEST(ParameterTest, IsSuitableFor)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(10));
	std::size_t const supported_size = description->element_size();

	turbodbc_test::mock_statement statement;

	turbodbc::parameter parameter(statement, parameter_index, 100, std::move(description));

	EXPECT_TRUE(parameter.is_suitable_for(turbodbc::type_code::string, supported_size));
	EXPECT_TRUE(parameter.is_suitable_for(turbodbc::type_code::string, supported_size - 1));

	EXPECT_FALSE(parameter.is_suitable_for(turbodbc::type_code::string, supported_size + 1));
	EXPECT_FALSE(parameter.is_suitable_for(turbodbc::type_code::integer, supported_size));
}

#include "turbodbc/column.h"

#include <gtest/gtest.h>
#include "mock_classes.h"

#include <turbodbc/descriptions/string_description.h>
#include <boost/variant/get.hpp>


namespace {
	void fill_buffer_with_value(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index, std::string const & value)
	{
		auto element = buffer[row_index];
		memcpy(element.data_pointer, value.data(), value.size() + 1);
		element.indicator = value.size();
	}

	void set_buffer_element_to_null(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index)
	{
		auto element = buffer[row_index];
		element.indicator = SQL_NULL_DATA;
	}

	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg2;
	}

	std::size_t const column_index = 42;

}

TEST(ColumnTest, GetFieldNonNullable)
{
	std::string const expected("this is a test string");
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(128));

	auto const buffer_type = description->column_c_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::column column(statement, column_index, buffered_rows, std::move(description));
	ASSERT_TRUE( buffer != nullptr);

	auto const row_index = 42;
	fill_buffer_with_value(*buffer, row_index, expected);
	EXPECT_EQ(expected, boost::get<std::string>(*column.get_field(row_index)));
}

TEST(ColumnTest, GetFieldNullable)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(128));

	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, testing::_, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::column column(statement, column_index, buffered_rows, std::move(description));
	ASSERT_TRUE( buffer != nullptr);

	auto const row_index = 42;
	set_buffer_element_to_null(*buffer, row_index);
	EXPECT_FALSE(static_cast<bool>(column.get_field(row_index)));
}

TEST(ColumnTest, GetInfo)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description("custom_name", false, 128));

	testing::NiceMock<turbodbc_test::mock_statement> statement;
	turbodbc::column column(statement, 0, 10, std::move(description));

	auto const info = column.get_info();
	EXPECT_EQ("custom_name", info.name);
	EXPECT_FALSE(info.supports_null_values);
	EXPECT_EQ(turbodbc::type_code::string, info.type);
}

TEST(ColumnTest, GetBuffer)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(128));

	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, testing::_, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	turbodbc::column column(statement, column_index, 100, std::move(description));
	EXPECT_EQ(buffer, &column.get_buffer());
}

TEST(ColumnTest, MoveConstructor)
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(128));

	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, testing::_, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	turbodbc::column moved(statement, column_index, 100, std::move(description));
	auto const expected_data_pointer = buffer->data_pointer();

	turbodbc::column column(std::move(moved));
	EXPECT_EQ(expected_data_pointer, column.get_buffer().data_pointer());
	EXPECT_EQ(nullptr, moved.get_buffer().data_pointer());
}

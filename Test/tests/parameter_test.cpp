#include "pydbc/parameter.h"

#include <cppunit/extensions/HelperMacros.h>

#include "mock_classes.h"

#include <pydbc/descriptions/integer_description.h>
#include <boost/variant/get.hpp>


class parameter_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( parameter_test );

	CPPUNIT_TEST( set_non_nullable );
//	CPPUNIT_TEST( get_field_nullable );

CPPUNIT_TEST_SUITE_END();

public:

	void set_non_nullable();
//	void get_field_nullable();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( parameter_test );

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
		*target_pointer = &arg3;
	}

	std::size_t const parameter_index = 42;

}

void parameter_test::set_non_nullable()
{
	std::unique_ptr<pydbc::integer_description> description(new pydbc::integer_description());

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	pydbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	pydbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	long const expected = 123;
	parameter.set(row_index, pydbc::field{expected});

	auto const actual = *reinterpret_cast<long *>((*buffer)[row_index].data_pointer);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

//void parameter_test::get_field_nullable()
//{
//	std::unique_ptr<pydbc::string_description> description(new pydbc::string_description(128));
//
//	pydbc_test::mock_statement statement;
//
//	cpp_odbc::multi_value_buffer * buffer = nullptr;
//	EXPECT_CALL(statement, do_bind_column(parameter_index, testing::_, testing::_))
//		.WillOnce(store_pointer_to_buffer_in(&buffer));
//
//	auto const buffered_rows = 100;
//	pydbc::column column(statement, parameter_index, buffered_rows, std::move(description));
//	CPPUNIT_ASSERT( buffer != nullptr);
//
//	auto const row_index = 42;
//	set_buffer_element_to_null(*buffer, row_index);
//	CPPUNIT_ASSERT(not static_cast<bool>(column.get_field(row_index)));
//}

#include "turbodbc/parameter_sets/set_field.h"

#include <gtest/gtest.h>
#include <tests/mock_classes.h>

#include <turbodbc/descriptions.h>



namespace {

std::size_t const param_index = 42;
std::size_t const n_params = 23;

}

using namespace turbodbc;
typedef turbodbc_test::mock_statement mock_statement;


TEST(SetFieldTest, ParameterIsSuitableForBoolean)
{
	mock_statement statement;
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));
	parameter const integer_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new integer_description()));

	field const value(true);
	EXPECT_TRUE(parameter_is_suitable_for(boolean_parameter, value));
	EXPECT_FALSE(parameter_is_suitable_for(integer_parameter, value));
}

TEST(SetFieldTest, ParameterIsSuitableForInteger)
{
	mock_statement statement;
	parameter const integer_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new integer_description()));
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));

	field const value(42l);
	EXPECT_TRUE(parameter_is_suitable_for(integer_parameter, value));
	EXPECT_FALSE(parameter_is_suitable_for(boolean_parameter, value));
}

TEST(SetFieldTest, ParameterIsSuitableForFloatingPoint)
{
	mock_statement statement;
	parameter const float_parameter(statement, param_index, n_params,
	                                std::unique_ptr<description>(new floating_point_description()));
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));

	field const value(3.14);
	EXPECT_TRUE(parameter_is_suitable_for(float_parameter, value));
	EXPECT_FALSE(parameter_is_suitable_for(boolean_parameter, value));
}

TEST(SetFieldTest, ParameterIsSuitableForTimestamp)
{
	mock_statement statement;
	parameter const ts_parameter(statement, param_index, n_params,
	                             std::unique_ptr<description>(new timestamp_description()));
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));

	field const value(boost::posix_time::ptime({2016, 9, 23}, {1, 2, 3}));
	EXPECT_TRUE(parameter_is_suitable_for(ts_parameter, value));
	EXPECT_FALSE(parameter_is_suitable_for(boolean_parameter, value));
}

TEST(SetFieldTest, ParameterIsSuitableForDate)
{
	mock_statement statement;
	parameter const date_parameter(statement, param_index, n_params,
	                               std::unique_ptr<description>(new date_description()));
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));

	field const value(boost::gregorian::date{2016, 9, 23});
	EXPECT_TRUE(parameter_is_suitable_for(date_parameter, value));
	EXPECT_FALSE(parameter_is_suitable_for(boolean_parameter, value));
}

TEST(SetFieldTest, ParameterIsSuitableForString)
{
	mock_statement statement;
	parameter const string_parameter(statement, param_index, n_params,
	                                 std::unique_ptr<description>(new string_description(7)));
	parameter const boolean_parameter(statement, param_index, n_params,
	                                  std::unique_ptr<description>(new boolean_description()));

	field const shorter(std::string("123456"));
	field const maximum_length(std::string("1234567"));
	field const too_long(std::string("12345678"));

	EXPECT_TRUE(parameter_is_suitable_for(string_parameter, shorter));
	EXPECT_TRUE(parameter_is_suitable_for(string_parameter, maximum_length));
	EXPECT_FALSE(parameter_is_suitable_for(string_parameter, too_long));
	EXPECT_FALSE(parameter_is_suitable_for(boolean_parameter, shorter));
}
#include "turbodbc/parameter_sets/bound_parameter_set.h"

#include <gtest/gtest.h>
#include <tests/mock_classes.h>

#include <stdexcept>


using namespace turbodbc;
typedef turbodbc_test::mock_statement mock_statement;


TEST(BoundParameterSetTest, ExecuteBatchThrowsIfBatchTooLarge)
{
	mock_statement statement;
	bound_parameter_set params(statement, 42);

	ASSERT_THROW(params.execute_batch(43), std::logic_error);
}


TEST(BoundParameterSetTest, TransferredSets)
{
	mock_statement statement;
	bound_parameter_set params(statement, 42);

	EXPECT_EQ(params.transferred_sets(), 0);
	params.execute_batch(17);
	EXPECT_EQ(params.transferred_sets(), 17);
	params.execute_batch(29);
	EXPECT_EQ(params.transferred_sets(), 46);
}

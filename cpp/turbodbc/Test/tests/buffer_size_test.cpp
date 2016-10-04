#include <turbodbc/buffer_size.h>

#include <gtest/gtest.h>

TEST(BufferSizeTest, SetRowsToValue)
{
    turbodbc::rows one_row(1);
    EXPECT_EQ(1, one_row.rows_to_buffer);
}

TEST(BufferSizeTest, SetMegabytesToValue)
{
    turbodbc::megabytes one_mb(1);
    EXPECT_EQ(1000000, one_mb.bytes_to_buffer);
}
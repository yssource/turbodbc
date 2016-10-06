#include <turbodbc/buffer_size.h>

#include <boost/variant.hpp>
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

TEST(BufferSizeTest, DetermineRowsToBuffer)
{
    turbodbc::buffer_size ten_rows(turbodbc::rows(10));
    std::size_t rows_to_buffer = boost::apply_visitor(turbodbc::determine_rows_to_buffer(), ten_rows);
    EXPECT_EQ(10, rows_to_buffer);
}
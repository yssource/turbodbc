#include <turbodbc/buffer_size.h>

namespace turbodbc {

rows::rows(std::size_t rows_to_buffer_):
    rows_to_buffer(rows_to_buffer_)
{
}

megabytes::megabytes(std::size_t megabytes_to_buffer) :
    bytes_to_buffer(megabytes_to_buffer * 1000000)
{
}

}
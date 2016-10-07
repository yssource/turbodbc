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

std::size_t determine_buffer_size::operator()(rows const& r) const
{
    return r.rows_to_buffer;
}

std::size_t determine_buffer_size::operator()(megabytes const& m) const
{
    return m.bytes_to_buffer;
}

}
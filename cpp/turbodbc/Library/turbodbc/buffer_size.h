#pragma once

#include <cstring>

namespace turbodbc {

struct rows {
    rows(std::size_t rows_to_buffer_);
    std::size_t rows_to_buffer;
};

struct megabytes {
    megabytes(std::size_t megabytes_to_buffer);
    std::size_t bytes_to_buffer;
};

}

turbodbc::rows r(42);
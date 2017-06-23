#pragma once

#include <cstdint>

namespace turbodbc {
    int64_t timestamp_to_microseconds(char const * data_pointer);
    intptr_t date_to_days(char const * data_pointer);
    void microseconds_to_timestamp(int64_t microseconds, char * data_pointer);
}

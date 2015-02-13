#pragma once

#include <cpp_odbc/column_description.h>
#include <pydbc/description.h>
#include <memory>

namespace pydbc {

std::unique_ptr<description const> make_description(cpp_odbc::column_description const & source);

}

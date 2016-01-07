#pragma once

#include <cpp_odbc/column_description.h>
#include <pydbc/description.h>
#include <pydbc/field.h>
#include <memory>

namespace pydbc {

/**
 * @brief Create a buffer description based on a given column description
 */
std::unique_ptr<description const> make_description(cpp_odbc::column_description const & source);

/**
 * @brief Create a buffer description based on the type and content of a value
 */
std::unique_ptr<description const> make_description(field const & value);


}

#pragma once

#include <boost/variant/variant.hpp>
#include <string>

namespace pydbc {

/**
 * @brief This type represents a single field in a table, i.e., the data associated
 *        with a given row and column
 */
using field = boost::variant<long, std::string>;

}

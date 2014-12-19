#pragma once

#include <boost/variant/variant.hpp>
#include <string>

namespace pydbc {

using field = boost::variant<long, std::string>;

}

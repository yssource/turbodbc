#pragma once

#include <pydbc/type_code.h>
#include <string>

namespace pydbc {

struct column_info {
	std::string name;
	type_code type;
	bool supports_null_values;
};

}

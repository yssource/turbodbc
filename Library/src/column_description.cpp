#include "cpp_odbc/column_description.h"

namespace cpp_odbc {

bool operator==(column_description const & lhs, column_description const & rhs)
{
	return (lhs.name == rhs.name) and (lhs.data_type == rhs.data_type)
			and (lhs.size == rhs.size) and (lhs.decimal_digits == rhs.decimal_digits)
			and (lhs.allows_null_values == rhs.allows_null_values);
}

}

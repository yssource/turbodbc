#pragma once

#include "sqltypes.h"

#include <string>

namespace cpp_odbc {


/**
 * @brief Information which fully describes a column of a result set
 */
struct column_description {
	std::string name;           ///< Column name
	SQLSMALLINT data_type;      ///< SQL data type constant
	SQLULEN size;               ///< Size of column. Corresponds to size of strings or precision of numeric types
	SQLSMALLINT decimal_digits; ///< Decimal digits of column. Corresponds to scale of numeric types and precision of timestamps
	bool allows_null_values;	///< True if NULL values are allowed in the column
};


}

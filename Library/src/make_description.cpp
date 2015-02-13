#include <pydbc/make_description.h>

#include <pydbc/description_types.h>
#include <sqlext.h>

#include <stdexcept>
#include <sstream>

namespace pydbc {

namespace {

	SQLULEN const digits_representable_by_long = 18;

	std::unique_ptr<description const> make_decimal_description(cpp_odbc::column_description const & source)
	{
		if (source.size <= digits_representable_by_long) {
			if (source.decimal_digits == 0) {
				return std::unique_ptr<description>(new integer_description);
			} else {
				return std::unique_ptr<description>(new floating_point_description);
			}
		} else {
			// fall back to strings; add two characters for decimal point and sign!
			return std::unique_ptr<description>(new string_description(source.size + 2));
		}
	}

}

std::unique_ptr<description const> make_description(cpp_odbc::column_description const & source)
{
	switch (source.data_type) {
		case SQL_CHAR:
		case SQL_VARCHAR:
		case SQL_LONGVARCHAR:
		case SQL_WVARCHAR:
		case SQL_WLONGVARCHAR:
		case SQL_WCHAR:
			return std::unique_ptr<description>(new string_description(source.size));
		case SQL_INTEGER:
		case SQL_SMALLINT:
		case SQL_BIGINT:
		case SQL_TINYINT:
			return std::unique_ptr<description>(new integer_description);
		case SQL_REAL:
		case SQL_FLOAT:
		case SQL_DOUBLE:
			return std::unique_ptr<description>(new floating_point_description);
		case SQL_BIT:
			return std::unique_ptr<description>(new boolean_description);
		case SQL_NUMERIC:
		case SQL_DECIMAL:
			return make_decimal_description(source);
		default:
			std::ostringstream message;
			message << "Error! Unsupported type identifier '" << source.data_type << "'";
			throw std::runtime_error(message.str());
	}
}

}

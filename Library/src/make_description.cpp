#include <pydbc/make_description.h>

#include <pydbc/description_types.h>
#include <sqlext.h>

#include <stdexcept>
#include <sstream>

namespace pydbc {

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
		case SQL_BIT:
			return std::unique_ptr<description>(new boolean_description);
		case SQL_DECIMAL:
			return std::unique_ptr<description>(new integer_description);
		default:
			std::ostringstream message;
			message << "Error! Unsupported type identifier '" << source.data_type << "'";
			throw std::runtime_error(message.str());
	}
}

}

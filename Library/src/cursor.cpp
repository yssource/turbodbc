/**
 *  @file cursor.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/cursor.h>
#include <sqlext.h>
#include <stdexcept>

namespace pydbc {

std::size_t const cached_rows = 10;

void cursor::execute(std::string const & sql)
{
	statement->execute(sql);
	if (statement->number_of_columns() != 0) {
		buffer = std::make_shared<cpp_odbc::multi_value_buffer>(sizeof(long), cached_rows);
		statement->bind_column(1, SQL_C_SBIGINT, *buffer);
	}
}

std::vector<long> cursor::fetch_one()
{
	if (buffer) {
		auto const has_results = statement->fetch_next();
		if (has_results) {
			auto value_ptr = reinterpret_cast<long *>((*buffer)[0].data_pointer);
			return {*value_ptr};
		} else {
			return {};
		}
	} else {
		throw std::runtime_error("No active result set");
	}
}

}

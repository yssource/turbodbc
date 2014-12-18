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

cursor::cursor(std::shared_ptr<cpp_odbc::statement> statement) :
	statement(statement)
{
}

void cursor::execute(std::string const & sql)
{
	statement->execute(sql);
	std::size_t const columns = statement->number_of_columns();
	if (columns != 0) {
		result = std::make_shared<result_set>(statement);
	}
}

std::vector<long> cursor::fetch_one()
{
	if (result) {
		return result->fetch_one();
	} else {
		throw std::runtime_error("No active result set");
	}
}

long cursor::get_rowcount()
{
	return statement->row_count();
}

}

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

cursor::cursor(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement)
{
}

void cursor::execute(std::string const & sql)
{
	statement_->execute(sql);
	std::size_t const columns = statement_->number_of_columns();
	if (columns != 0) {
		result_ = std::make_shared<result_set>(statement_, 10);
	}
}

std::vector<nullable_field> cursor::fetch_one()
{
	if (result_) {
		return result_->fetch_one();
	} else {
		throw std::runtime_error("No active result set");
	}
}

long cursor::get_rowcount()
{
	return statement_->row_count();
}

std::shared_ptr<cpp_odbc::statement const> cursor::get_statement() const
{
	return statement_;
}

}

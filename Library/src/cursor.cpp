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
#include <pydbc/make_description.h>

#include <cpp_odbc/statement.h>
#include <cpp_odbc/error.h>

#include <boost/variant/get.hpp>
#include <sqlext.h>
#include <stdexcept>

#include <cstring>
#include <sstream>


namespace pydbc {

cursor::cursor(std::shared_ptr<cpp_odbc::connection const> connection) :
	connection_(connection),
	query_()
{
}

cursor::~cursor() = default;

void cursor::prepare(std::string const & sql)
{
	query_.reset();
	auto statement = connection_->make_statement();
	statement->prepare(sql);
	query_ = std::make_shared<query>(statement);
}

void cursor::execute()
{
	query_->execute();
}

void cursor::bind_parameters()
{
	query_->bind_parameters();
}

void cursor::add_parameter_set(std::vector<nullable_field> const & parameter_set)
{
	query_->add_parameter_set(parameter_set);
}

std::vector<nullable_field> cursor::fetch_one()
{
	return query_->fetch_one();
}

long cursor::get_row_count()
{
	return query_->get_row_count();
}

std::shared_ptr<cpp_odbc::connection const> cursor::get_connection() const
{
	return connection_;
}


}

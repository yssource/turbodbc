/**
 *  @file connection.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "pydbc/connection.h"

namespace pydbc {

connection::connection(std::shared_ptr<cpp_odbc::connection> low_level_connection) :
	connection_(low_level_connection)
{
}

void connection::commit()
{
	connection_->commit();
}

cursor connection::make_cursor()
{
	return {connection_->make_statement()};
}

}

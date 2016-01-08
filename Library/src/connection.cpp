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
#include <sqlext.h>

namespace pydbc {

connection::connection(std::shared_ptr<cpp_odbc::connection const> low_level_connection) :
	rows_to_buffer(1000),
	parameter_sets_to_buffer(1000),
	connection_(low_level_connection)
{
	connection_->set_attribute(SQL_ATTR_AUTOCOMMIT, SQL_AUTOCOMMIT_OFF);
}

void connection::commit() const
{
	connection_->commit();
}

void connection::rollback() const
{
	connection_->rollback();
}

cursor connection::make_cursor() const
{
	return {connection_};
}

}

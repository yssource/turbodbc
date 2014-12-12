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

void py_connection::commit()
{
	connection->commit();
}

py_cursor py_connection::cursor()
{
	return {psapp::to_valid(connection->make_statement())};
}

}

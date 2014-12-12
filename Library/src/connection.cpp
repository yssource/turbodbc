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

void connection::commit()
{
	connection->commit();
}

cursor connection::make_cursor()
{
	return {psapp::to_valid(connection->make_statement())};
}

}

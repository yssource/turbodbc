#pragma once
/**
 *  @file connection.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "pydbc/cursor.h"
#include <cpp_odbc/connection.h>
#include <psapp/valid_ptr.h>

namespace pydbc {

struct connection {
	psapp::valid_ptr<cpp_odbc::connection> connection;

	void commit();

	pydbc::cursor make_cursor();
};

}

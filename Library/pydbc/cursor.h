#pragma once
/**
 *  @file cursor.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <cpp_odbc/statement.h>
#include <psapp/valid_ptr.h>

namespace pydbc {

struct py_cursor {
	psapp::valid_ptr<cpp_odbc::statement> statement;

	void execute(std::string const & sql);
};

}

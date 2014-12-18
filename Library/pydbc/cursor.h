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
#include <cpp_odbc/multi_value_buffer.h>
#include <memory>
#include <vector>

namespace pydbc {

struct cursor {
	std::shared_ptr<cpp_odbc::statement> statement;
	std::shared_ptr<cpp_odbc::multi_value_buffer> buffer;

	void execute(std::string const & sql);
	std::vector<int> fetch_one();
};

}

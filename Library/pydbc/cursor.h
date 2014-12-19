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
#include <pydbc/result_set.h>
#include <memory>
#include <vector>

namespace pydbc {

struct cursor {
	std::shared_ptr<cpp_odbc::statement> statement;
	std::shared_ptr<result_set> result;

	cursor(std::shared_ptr<cpp_odbc::statement> statement);

	void execute(std::string const & sql);
	std::vector<field> fetch_one();
	long get_rowcount();
};

}

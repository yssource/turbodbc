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

struct result_set {
	std::vector<cpp_odbc::multi_value_buffer> columns;

	result_set(std::size_t number_of_columns);
};

struct cursor {
	std::shared_ptr<cpp_odbc::statement> statement;
	std::shared_ptr<result_set> result;

	cursor(std::shared_ptr<cpp_odbc::statement> statement);

	void execute(std::string const & sql);
	std::vector<long> fetch_one();
};

}

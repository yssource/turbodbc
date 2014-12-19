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
#include <pydbc/column.h>
#include <vector>
#include <memory>

namespace pydbc {

struct result_set {
	std::shared_ptr<cpp_odbc::statement> statement;
	std::vector<std::unique_ptr<column>> columns;

	result_set(std::shared_ptr<cpp_odbc::statement> statement);

	std::vector<field> fetch_one();
};

}

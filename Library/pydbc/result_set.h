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

class result_set {
public:
	result_set(std::shared_ptr<cpp_odbc::statement const> statement);

	std::vector<field> fetch_one();
private:
	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::vector<std::unique_ptr<column>> columns_;
};

}

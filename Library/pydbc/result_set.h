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

/**
 * @brief This class represents a result set as created by database queries.
 */
class result_set {
public:
	/**
	 * @brief Create a new result set. All necessary data structures and buffers
	 *        are created and bound to the given statement
	 */
	result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows);

	/**
	 * @brief Fetch the next row of the result set
	 */
	std::vector<nullable_field> fetch_one();
private:
	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::vector<std::unique_ptr<column>> columns_;
	std::size_t rows_fetched_;
	std::size_t current_fetched_row_;
};

}

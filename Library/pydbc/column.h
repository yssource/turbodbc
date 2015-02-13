#pragma once

#include <cpp_odbc/statement.h>
#include <pydbc/field.h>
#include <pydbc/description.h>

namespace pydbc {

/**
 * @brief This interface represents a single column of a result set.
 */
class column {
public:
	/**
	 * @brief Create a new column, binding an internal buffer to the statement
	 * @param statement The statement for which to bind a buffer
	 * @param one_based_index One-based column index for bind command
	 * @param desription Description concerning data type of column
	 */
	column(cpp_odbc::statement const & statement, std::size_t one_based_index, std::unique_ptr<description const> description);

	/**
	 * @brief Retrieve the field of the current result set row associated with this column
	 */
	nullable_field get_field() const;

	~column();
private:
	std::unique_ptr<description const> description_;
	cpp_odbc::multi_value_buffer buffer_;
};


}

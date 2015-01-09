#pragma once

#include <pydbc/column.h>
#include <cpp_odbc/statement.h>
#include <cpp_odbc/multi_value_buffer.h>

namespace pydbc {

std::size_t const cached_rows = 10;

/**
 * @brief This concrete class represents a column of integers
 */
struct long_column : public column {
	long_column(cpp_odbc::statement const & statement, std::size_t one_based_index);
private:
	field do_get_field() const final;
	cpp_odbc::multi_value_buffer buffer_;
};

/**
 * @brief This concrete class represents a column of strings
 */
struct string_column : public column {
	string_column(cpp_odbc::statement const & statement, std::size_t one_based_index);
private:
	field do_get_field() const final;
	cpp_odbc::multi_value_buffer buffer_;
};


}

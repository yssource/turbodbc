/**
 *  @file cursor.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/cursor.h>
#include <sqlext.h>
#include <stdexcept>

namespace pydbc {

std::size_t const cached_rows = 10;

result_set::result_set(std::shared_ptr<cpp_odbc::statement> statement) :
	statement(statement)
{
	std::size_t const n_columns = statement->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		columns.emplace_back(sizeof(long), cached_rows);
		auto & new_column = columns.back();
		statement->bind_column(one_based_index, SQL_C_SBIGINT, new_column);
	}
}

std::vector<long> result_set::fetch_one()
{
	auto const has_results = statement->fetch_next();
	if (has_results) {
		std::vector<long> row;
		for (auto const & column : columns) {
			auto value_ptr = reinterpret_cast<long const *>(column[0].data_pointer);
			row.push_back(*value_ptr);
		}
		return row;
	} else {
		return {};
	}
}


}

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

result_set::result_set(std::shared_ptr<cpp_odbc::statement> statement) :
	statement(statement)
{
	std::size_t const n_columns = statement->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		columns.emplace_back(new long_column(*statement, one_based_index));
	}
}

std::vector<field> result_set::fetch_one()
{
	auto const has_results = statement->fetch_next();
	if (has_results) {
		std::vector<field> row;
		for (auto const & column : columns) {
			row.push_back(column->get_field());
		}
		return row;
	} else {
		return {};
	}
}


}

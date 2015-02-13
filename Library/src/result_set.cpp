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
#include <pydbc/column.h>
#include <pydbc/make_description.h>
#include <sqlext.h>
#include <stdexcept>
#include <sstream>

namespace pydbc {

namespace {

	std::unique_ptr<column> make_column(cpp_odbc::statement const & statement, std::size_t one_based_index)
	{
		auto const column_description = statement.describe_column(one_based_index);
		return std::unique_ptr<column>(new column(statement, one_based_index, make_description(column_description)));
	}

}

result_set::result_set(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement),
	rows_fetched_(0)
{
	std::size_t const n_columns = statement_->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		columns_.push_back(make_column(*statement, one_based_index));
	}

	statement_->set_statement_attribute(SQL_ATTR_ROW_ARRAY_SIZE, 1);
}

std::vector<nullable_field> result_set::fetch_one()
{
	auto const has_results = statement_->fetch_next();
	if (has_results) {
		std::vector<nullable_field> row;
		for (auto const & column : columns_) {
			row.push_back(column->get_field());
		}
		return row;
	} else {
		return {};
	}
}


}

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
#include <pydbc/column_types.h>
#include <sqlext.h>
#include <stdexcept>
#include <sstream>

namespace pydbc {

namespace {

	std::unique_ptr<column> make_column(cpp_odbc::statement const & statement, std::size_t one_based_index)
	{
		auto const description = statement.describe_column(one_based_index);

		switch (description.data_type) {
			case SQL_VARCHAR:
			case SQL_LONGVARCHAR:
			case SQL_WVARCHAR:
			case SQL_CHAR:
			case SQL_WLONGVARCHAR:
			case SQL_WCHAR:
				return std::unique_ptr<column>(new string_column(statement, one_based_index));
			case SQL_INTEGER:
			case SQL_SMALLINT:
			case SQL_BIGINT:
			case SQL_BIT:
				return std::unique_ptr<column>(new long_column(statement, one_based_index));
			case SQL_DECIMAL:
				return std::unique_ptr<column>(new long_column(statement, one_based_index));
			default:
				std::ostringstream message;
				message << "Error! Unsupported type identifier '" << description.data_type << "'";
				throw std::runtime_error(message.str());
		}
	}

}

result_set::result_set(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement)
{
	std::size_t const n_columns = statement_->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		columns_.push_back(make_column(*statement, one_based_index));
	}
}

std::vector<field> result_set::fetch_one()
{
	auto const has_results = statement_->fetch_next();
	if (has_results) {
		std::vector<field> row;
		for (auto const & column : columns_) {
			row.push_back(*column->get_field());
		}
		return row;
	} else {
		return {};
	}
}


}

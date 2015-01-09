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

namespace {

	bool is_string_type(long type)
	{
		switch (type) {
			case SQL_VARCHAR:
			case SQL_LONGVARCHAR:
			case SQL_WVARCHAR:
			case SQL_CHAR:
			case SQL_WLONGVARCHAR:
			case SQL_WCHAR:
				return true;
			default:
				return false;
		}

	}

}

result_set::result_set(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement)
{
	std::size_t const n_columns = statement_->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		auto const type = statement_->get_integer_column_attribute(one_based_index, SQL_DESC_TYPE);

		if (is_string_type(type)) {
			columns_.emplace_back(new string_column(*statement_, one_based_index));
		} else {
			columns_.emplace_back(new long_column(*statement_, one_based_index));
		}
	}
}

std::vector<field> result_set::fetch_one()
{
	auto const has_results = statement_->fetch_next();
	if (has_results) {
		std::vector<field> row;
		for (auto const & column : columns_) {
			row.push_back(column->get_field());
		}
		return row;
	} else {
		return {};
	}
}


}

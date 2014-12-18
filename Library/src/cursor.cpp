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

cursor::cursor(std::shared_ptr<cpp_odbc::statement> statement) :
	statement(statement)
{
}

void cursor::execute(std::string const & sql)
{
	statement->execute(sql);
	std::size_t const columns = statement->number_of_columns();
	if (columns != 0) {
		result = std::make_shared<result_set>(columns);

		for (std::size_t zero_based_index = 0; zero_based_index != columns; ++zero_based_index) {
			auto const one_based_index = zero_based_index + 1;
			statement->bind_column(one_based_index, SQL_C_SBIGINT, result->columns[zero_based_index]);
		}
	}
}

std::vector<long> cursor::fetch_one()
{
	if (result) {
		auto const has_results = statement->fetch_next();
		if (has_results) {
			std::vector<long> row;
			for (auto const & column : result->columns) {
				auto value_ptr = reinterpret_cast<long const *>(column[0].data_pointer);
				row.push_back(*value_ptr);
			}
			return row;
		} else {
			return {};
		}
	} else {
		throw std::runtime_error("No active result set");
	}
}

}

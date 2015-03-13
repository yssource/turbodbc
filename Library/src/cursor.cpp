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

#include <cstring>

namespace pydbc {

cursor::cursor(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement)
{
}

void cursor::prepare(std::string const & sql)
{
	statement_->prepare(sql);
}

void cursor::execute()
{
	statement_->execute_prepared();

	std::size_t const columns = statement_->number_of_columns();
	if (columns != 0) {
		result_ = std::make_shared<result_set>(statement_, 10);
	}
}

void cursor::execute_many()
{
	std::vector<cpp_odbc::multi_value_buffer> parameters;
	if (statement_->number_of_parameters() != 0) {

		for (SQLSMALLINT p = 0; p != statement_->number_of_parameters(); ++p) {
			parameters.emplace_back(sizeof(long), 10);
			statement_->bind_input_parameter(p + 1, SQL_C_SBIGINT, SQL_BIGINT, parameters.back());
		}

		for (unsigned int i = 0; i != 3; ++i) {
			auto element = parameters[0][i];
			long value = i + 1;
			memcpy(element.data_pointer, &value, sizeof(value));
			element.indicator = sizeof(value);
		}
		statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, 3l);
	}

	statement_->execute_prepared();

	std::size_t const columns = statement_->number_of_columns();
	if (columns != 0) {
		result_ = std::make_shared<result_set>(statement_, 10);
	}
}

std::vector<nullable_field> cursor::fetch_one()
{
	if (result_) {
		return result_->fetch_one();
	} else {
		throw std::runtime_error("No active result set");
	}
}

long cursor::get_rowcount()
{
	return statement_->row_count();
}

std::shared_ptr<cpp_odbc::statement const> cursor::get_statement() const
{
	return statement_;
}

}

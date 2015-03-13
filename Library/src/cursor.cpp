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

#include <boost/variant/get.hpp>
#include <sqlext.h>
#include <stdexcept>

#include <cstring>

namespace pydbc {

cursor::cursor(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement),
	current_parameter_set_(0)
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

void cursor::bind_parameters()
{
	if (statement_->number_of_parameters() != 0) {
		parameters_ = std::make_shared<std::vector<cpp_odbc::multi_value_buffer>>();

		for (SQLSMALLINT p = 0; p != statement_->number_of_parameters(); ++p) {
			parameters_->emplace_back(sizeof(long), 10);
			statement_->bind_input_parameter(p + 1, SQL_C_SBIGINT, SQL_BIGINT, parameters_->back());
		}
	}
}

void cursor::add_parameter_set(std::vector<nullable_field> const & parameter_set)
{
	for (unsigned int parameter = 0; parameter != parameter_set.size(); ++parameter) {
		auto element = (*parameters_)[parameter][current_parameter_set_];
		auto value = boost::get<long>(*parameter_set[parameter]);
		memcpy(element.data_pointer, &value, sizeof(value));
		element.indicator = sizeof(value);
	}
	++current_parameter_set_;
	statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, current_parameter_set_);
}

void cursor::execute_many()
{
	bind_parameters();

	for (unsigned int i = 0; i != 3; ++i) {
		add_parameter_set({{pydbc::field(static_cast<long>(i + 1))}});
	}

	execute();
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

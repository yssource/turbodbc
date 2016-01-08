#include <pydbc/query.h>
#include <pydbc/make_description.h>
#include <pydbc/descriptions/integer_description.h>

#include <cpp_odbc/error.h>

#include <boost/variant/get.hpp>
#include <sqlext.h>
#include <stdexcept>

#include <cstring>
#include <sstream>


namespace pydbc {

namespace {

	std::shared_ptr<parameter> make_parameter(cpp_odbc::statement const & statement, std::size_t one_based_index)
	{
		auto const description = statement.describe_parameter(one_based_index);
		return std::make_shared<parameter>(statement, one_based_index, 10, make_description(description));
	}

}

query::query(std::shared_ptr<cpp_odbc::statement const> statement) :
	statement_(statement),
	current_parameter_set_(0),
	was_executed_(false)
{
	bind_parameters();
}

query::~query()
{
	statement_->close_cursor();
}

void query::execute()
{
	execute_batch();

	std::size_t const columns = statement_->number_of_columns();
	if (columns != 0) {
		result_ = std::make_shared<result_set>(statement_, 10);
	}
}

void query::add_parameter_set(std::vector<nullable_field> const & parameter_set)
{
	check_parameter_set(parameter_set);

	for (unsigned int p = 0; p != parameter_set.size(); ++p) {
		add_parameter(p, parameter_set[p]);
	}

	++current_parameter_set_;
}

std::vector<nullable_field> query::fetch_one()
{
	if (result_) {
		return result_->fetch_one();
	} else {
		throw std::runtime_error("No active result set");
	}
}

long query::get_row_count()
{
	if (was_executed_) {
		return statement_->row_count();
	} else {
		return 0;
	}
}

void query::execute_batch()
{
	if (current_parameter_set_ != 0) {
		if (parameters_.size() != 0) {
			statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, current_parameter_set_);
		}
		statement_->execute_prepared();
		was_executed_ = true;
	} else {
		if (parameters_.size() == 0) {
			statement_->execute_prepared();
			was_executed_ = true;
		}
	}
}

void query::bind_parameters()
{
	if (statement_->number_of_parameters() != 0) {
		std::size_t const n_parameters = statement_->number_of_parameters();
		for (std::size_t one_based_index = 1; one_based_index <= n_parameters; ++one_based_index) {
			parameters_.push_back(make_parameter(*statement_, one_based_index));
		}
		statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, current_parameter_set_);
	}
}

void query::check_parameter_set(std::vector<nullable_field> const & parameter_set) const
{
	if (parameter_set.size() != parameters_.size()) {
		std::ostringstream message;
		message << "Invalid number of parameters (expected " << parameters_.size()
				<< ", got " << parameter_set.size() << ")";
		throw cpp_odbc::error(message.str());
	}
}

void query::add_parameter(std::size_t index, nullable_field const & value)
{
	try {
		parameters_[index]->set(current_parameter_set_, value);
	} catch (boost::bad_get const &) {
		execute_batch();
		recover_unwritten_parameters_below(index);
		rebind_parameter_to_hold_value(index, *value);
		parameters_[index]->set(current_parameter_set_, value);
	}
}

void query::recover_unwritten_parameters_below(std::size_t index)
{
	for (std::size_t i = 0; i != index; ++i) {
		parameters_[i]->copy_to_first_row(current_parameter_set_);
	}
	current_parameter_set_ = 0;
}

void query::rebind_parameter_to_hold_value(std::size_t index, field const & value)
{
	auto description = make_description(value);
	parameters_[index] = std::make_shared<parameter>(*statement_, index + 1, 10, std::move(description));
}

}

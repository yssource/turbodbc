#include <turbodbc/parameter_sets/field_parameter_set.h>
#include <turbodbc/parameter_sets/set_field.h>
#include <turbodbc/make_description.h>

#include <cpp_odbc/error.h>

#include <sqlext.h>

#include <sstream>


namespace turbodbc {

namespace {

	std::shared_ptr<parameter> make_parameter(cpp_odbc::statement const & statement, std::size_t one_based_index, std::size_t buffered_sets)
	{
		auto description = make_description(statement.describe_parameter(one_based_index));
		if ((description->get_type_code() == type_code::string) and (description->element_size() > 51)) {
			auto modified_description = statement.describe_parameter(one_based_index);
			modified_description.size = 50;
			description = make_description(modified_description);
		}
		return std::make_shared<parameter>(statement, one_based_index, buffered_sets, std::move(description));
	}

}

field_parameter_set::field_parameter_set(std::shared_ptr<cpp_odbc::statement const> statement,
                                         std::size_t parameter_sets_to_buffer) :
	statement_(statement),
	parameter_sets_to_buffer_(parameter_sets_to_buffer),
	current_parameter_set_(0),
	row_count_(0),
	rows_processed_(0)
{
	bind_parameters();
}

field_parameter_set::~field_parameter_set() = default;

void field_parameter_set::flush()
{
	execute_batch();
}

void field_parameter_set::add_parameter_set(std::vector<nullable_field> const & parameter_set)
{
	check_parameter_set(parameter_set);

	if (current_parameter_set_ == parameter_sets_to_buffer_) {
		execute_batch();
	}

	for (unsigned int p = 0; p != parameter_set.size(); ++p) {
		add_parameter(p, parameter_set[p]);
	}

	++current_parameter_set_;
}

long field_parameter_set::get_row_count()
{
	return row_count_;
}

std::size_t field_parameter_set::execute_batch()
{
	std::size_t result = 0;

	if (not parameters_.empty()) {
		statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, current_parameter_set_);
	}

	if ((current_parameter_set_ != 0) or parameters_.empty()){
		statement_->execute_prepared();
		update_row_count();
		result = parameters_.empty() ? 1 : current_parameter_set_;
	}

	current_parameter_set_ = 0;
	return result;
}

void field_parameter_set::bind_parameters()
{
	if (statement_->number_of_parameters() != 0) {
		std::size_t const n_parameters = statement_->number_of_parameters();
		for (std::size_t one_based_index = 1; one_based_index <= n_parameters; ++one_based_index) {
			parameters_.push_back(make_parameter(*statement_, one_based_index, parameter_sets_to_buffer_));
		}
		statement_->set_attribute(SQL_ATTR_PARAMSET_SIZE, current_parameter_set_);
		statement_->set_attribute(SQL_ATTR_PARAMS_PROCESSED_PTR, &rows_processed_);
	}
}

void field_parameter_set::check_parameter_set(std::vector<nullable_field> const & parameter_set) const
{
	if (parameter_set.size() != parameters_.size()) {
		std::ostringstream message;
		message << "Invalid number of parameters (expected " << parameters_.size()
				<< ", got " << parameter_set.size() << ")";
		throw cpp_odbc::error(message.str());
	}
}

void field_parameter_set::add_parameter(std::size_t index, nullable_field const & value)
{
	if (value) {
		if (parameter_is_suitable_for(*parameters_[index], *value)) {
			auto element = parameters_[index]->get_buffer()[current_parameter_set_];
			set_field(*value, element);
		} else {
			auto const last_active_row = execute_batch();
			recover_unwritten_parameters_below(index, last_active_row);
			rebind_parameter_to_hold_value(index, *value);
			auto element = parameters_[index]->get_buffer()[current_parameter_set_];
			set_field(*value, element);
		}
	} else {
		auto element = parameters_[index]->get_buffer()[current_parameter_set_];
		set_null(element);
	}
}

void field_parameter_set::recover_unwritten_parameters_below(std::size_t parameter_index, std::size_t last_active_row)
{
	for (std::size_t i = 0; i != parameter_index; ++i) {
		move_to_top(*parameters_[i], last_active_row);
	}
}

void field_parameter_set::rebind_parameter_to_hold_value(std::size_t index, field const & value)
{
	auto description = make_description(value);
	parameters_[index] = std::make_shared<parameter>(*statement_, index + 1, parameter_sets_to_buffer_, std::move(description));
}

void field_parameter_set::update_row_count()
{
	row_count_ += rows_processed_;
}

}

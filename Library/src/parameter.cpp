#include <pydbc/parameter.h>

namespace pydbc {

parameter::parameter(cpp_odbc::statement const & statement, std::size_t one_based_index, std::size_t buffered_rows, std::unique_ptr<description const> description) :
	description_(std::move(description)),
	buffer_(description_->element_size(), buffered_rows)
{
	statement.bind_input_parameter(one_based_index, description_->column_c_type(), description_->column_sql_type(), buffer_);
}

parameter::~parameter() = default;

void parameter::set(std::size_t row_index, pydbc::nullable_field const & value)
{
	auto element = buffer_[row_index];
	if (value) {
		description_->set_field(element, *value);
	} else {
		element.indicator = SQL_NULL_DATA;
	}
}


}

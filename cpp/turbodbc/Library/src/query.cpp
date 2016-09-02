#include <turbodbc/query.h>
#include <turbodbc/make_description.h>
#include <turbodbc/descriptions/integer_description.h>
#include <turbodbc/result_sets/bound_result_set.h>
#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <cpp_odbc/error.h>

#include <boost/variant/get.hpp>
#include <sqlext.h>
#include <stdexcept>

#include <cstring>
#include <sstream>


namespace turbodbc {

query::query(std::shared_ptr<cpp_odbc::statement const> statement,
             std::size_t rows_to_buffer,
             std::size_t parameter_sets_to_buffer,
             bool use_double_buffering) :
	statement_(statement),
	parameters_(statement, parameter_sets_to_buffer),
	rows_to_buffer_(rows_to_buffer),
	use_double_buffering_(use_double_buffering)
{
}

query::~query()
{
	results_.reset(); // result may access statement concurrently!
	statement_->close_cursor();
}

void query::execute()
{
	parameters_.flush();

	std::size_t const columns = statement_->number_of_columns();
	if (columns != 0) {
		if (use_double_buffering_) {
			results_ = std::make_shared<result_sets::double_buffered_result_set>(statement_, rows_to_buffer_);
		} else {
			results_ = std::make_shared<result_sets::bound_result_set>(statement_, rows_to_buffer_);
		}
	}
}

std::shared_ptr<turbodbc::result_sets::result_set> query::get_results()
{
	return results_;
}

void query::add_parameter_set(std::vector<nullable_field> const & parameter_set)
{
	parameters_.add_parameter_set(parameter_set);
}

long query::get_row_count()
{
	bool const has_result_set = (statement_->number_of_columns() != 0);
	if (has_result_set) {
		return statement_->row_count();
	} else {
		return parameters_.get_row_count();
	}
}

}

#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <turbodbc/make_description.h>

#include <sqlext.h>

namespace turbodbc { namespace result_sets {

double_buffered_result_set::double_buffered_result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows) :
	statement_(statement)
{
	std::size_t const n_columns = statement_->number_of_columns();

	for (std::size_t one_based_index = 1; one_based_index <= n_columns; ++one_based_index) {
		auto column_description = make_description(statement_->describe_column(one_based_index));
		batches_[0].columns.emplace_back(*statement, one_based_index, buffered_rows, std::move(column_description));
	}

	auto rows_per_single_buffer = buffered_rows / 2 + buffered_rows % 2;
	statement_->set_attribute(SQL_ATTR_ROW_ARRAY_SIZE, rows_per_single_buffer);
	statement_->set_attribute(SQL_ATTR_ROWS_FETCHED_PTR, &(batches_[0].rows_fetched));
}

double_buffered_result_set::~double_buffered_result_set() = default;

std::size_t double_buffered_result_set::do_fetch_next_batch()
{
	statement_->fetch_next();
	return batches_[0].rows_fetched;
}


std::vector<column_info> double_buffered_result_set::do_get_column_info() const
{
	std::vector<column_info> infos;
	for (auto const & column : batches_[0].columns) {
		infos.push_back(column.get_info());
	}
	return infos;
}


std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> double_buffered_result_set::do_get_buffers() const
{
	throw 42;
//	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> buffers;
//	for (auto const & column : columns_) {
//		buffers.push_back(std::cref(column.get_buffer()));
//	}
//	return buffers;
}


} }

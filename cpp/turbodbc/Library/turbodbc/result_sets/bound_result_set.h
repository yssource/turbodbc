#pragma once

#include <turbodbc/result_sets/result_set.h>
#include <turbodbc/column.h>

#include <cpp_odbc/statement.h>
#include <memory>


namespace turbodbc { namespace result_sets {

class bound_result_set : public turbodbc::result_sets::result_set {
public:
	bound_result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows);
	virtual ~bound_result_set();
private:
	std::size_t do_fetch_next_batch() final;
	std::vector<column_info> do_get_column_info() const final;
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> do_get_buffers() const final;

	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::vector<column> columns_;
	std::size_t rows_fetched_;
};


} }

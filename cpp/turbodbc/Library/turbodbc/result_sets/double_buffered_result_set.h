#pragma once

#include <turbodbc/result_sets/result_set.h>
#include <turbodbc/column.h>

#include <cpp_odbc/statement.h>
#include <memory>
#include <array>


namespace turbodbc { namespace result_sets {

/**
 * @brief This class implements result_set by double buffering real ODBC
 *        result sets. This means that while one buffer is filled by the database,
 *        users retrieve values from a previously filled buffer.
 */
class double_buffered_result_set : public turbodbc::result_sets::result_set {
public:
	/**
	 * @brief Prepare and bind buffers suitable of holding buffered_rows to
	 *        the given statement.
	 */
	double_buffered_result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows);
	virtual ~double_buffered_result_set();
private:
	std::size_t do_fetch_next_batch() final;
	std::vector<column_info> do_get_column_info() const final;
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> do_get_buffers() const final;

	struct batch_buffer {
		std::vector<column> columns;
		std::size_t rows_fetched;
	};

	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::array<batch_buffer, 2> batches_;
	std::size_t active_reading_batch_;
};


} }

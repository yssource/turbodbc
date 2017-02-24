#pragma once

#include <turbodbc/buffer_size.h>

#include <cpp_odbc/connection.h>
#include <turbodbc/command.h>
#include <turbodbc/column_info.h>
#include <turbodbc/result_sets/result_set.h>
#include <memory>


namespace turbodbc {

/**
 * TODO: Cursor needs proper unit tests
 */
class cursor {
public:
	cursor(std::shared_ptr<cpp_odbc::connection const> connection,
		   turbodbc::buffer_size buffer_size,
		   std::size_t parameter_sets_to_buffer,
		   bool use_async_io,
		   bool query_db_for_parameter_types);

	void prepare(std::string const & sql);
	void execute();
	void add_parameter_set(std::vector<nullable_field> const & parameter_set);

	long get_row_count();

	std::shared_ptr<result_sets::result_set> get_result_set() const;

	std::shared_ptr<cpp_odbc::connection const> get_connection() const;

	std::shared_ptr<turbodbc::command> get_command();

	~cursor();

private:
	std::shared_ptr<cpp_odbc::connection const> connection_;
	turbodbc::buffer_size buffer_size_;
	std::size_t parameter_sets_to_buffer_;
	bool use_async_io_;
	bool query_db_for_parameter_types_;
	std::shared_ptr<turbodbc::command> command_;
	std::shared_ptr<result_sets::result_set> results_;
};

}

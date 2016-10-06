#pragma once

#include <turbodbc/buffer_size.h>

#include <cpp_odbc/statement.h>
#include <turbodbc/result_sets/result_set.h>
#include <turbodbc/parameter_sets/field_parameter_set.h>
#include <memory>


namespace turbodbc {


class query {
public:
	query(std::shared_ptr<cpp_odbc::statement const> statement,
		  turbodbc::buffer_size buffer_size,
		  std::size_t parameter_sets_to_buffer,
		  bool use_double_buffering);

	void execute();
	std::shared_ptr<turbodbc::result_sets::result_set> get_results();
	void add_parameter_set(std::vector<nullable_field> const & parameter_set);

	long get_row_count();

	~query();

private:
	std::shared_ptr<cpp_odbc::statement const> statement_;
	field_parameter_set parameters_;
	turbodbc::buffer_size buffer_size_;
	bool use_double_buffering_;
	std::shared_ptr<result_sets::result_set> results_;
};

}

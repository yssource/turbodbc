#pragma once

#include <cpp_odbc/statement.h>
#include <turbodbc/parameter.h>

#include <vector>


namespace turbodbc {


class bound_parameter_set {
public:
	bound_parameter_set(cpp_odbc::statement const & statement,
	                    std::size_t buffered_sets);

	std::size_t transferred_sets() const;

	std::vector<std::shared_ptr<parameter>> const & get_parameters();

	void execute_batch(std::size_t sets_in_batch);
private:
	cpp_odbc::statement const & statement_;
	std::size_t buffered_sets_;
	std::size_t transferred_sets_;
	SQLULEN confirmed_last_batch_;
	std::vector<std::shared_ptr<parameter>> parameters_;
};


}
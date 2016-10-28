#pragma once

#include <cpp_odbc/statement.h>

namespace turbodbc {


class bound_parameter_set {
public:
	bound_parameter_set(cpp_odbc::statement const & statement,
	                    std::size_t buffered_sets);

	std::size_t transferred_sets() const;

	void execute_batch(std::size_t sets_in_batch);
private:
	std::size_t buffered_sets_;
	std::size_t transferred_sets_;
};


}
#include <turbodbc/parameter_sets/bound_parameter_set.h>

#include <stdexcept>

namespace turbodbc {


bound_parameter_set::bound_parameter_set(cpp_odbc::statement const &,
                                         std::size_t buffered_sets) :
		buffered_sets_(buffered_sets),
		transferred_sets_(0)
{}

std::size_t bound_parameter_set::transferred_sets() const
{
	return transferred_sets_;
}

void bound_parameter_set::execute_batch(std::size_t sets_in_batch)
{
	if (sets_in_batch <= buffered_sets_) {
		transferred_sets_ += sets_in_batch;
	} else {
		throw std::logic_error("A batch cannot be larger than the number of buffered sets");
	}
}


}
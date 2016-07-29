#pragma once

#include <turbodbc_numpy/masked_column.h>
#include <turbodbc_numpy/numpy_type.h>

namespace turbodbc_numpy {

class datetime_column : public masked_column {
public:
	datetime_column();
	virtual ~datetime_column();
private:
	void do_append(cpp_odbc::multi_value_buffer const & buffer, std::size_t n_values) final;

	boost::python::object do_get_data() final;
	boost::python::object do_get_mask() final;

	void resize(std::size_t new_size);

	boost::python::object data_;
	boost::python::object mask_;
	std::size_t size_;
};

}

#include <turbodbc/buffer_size.h>

#include <boost/python/class.hpp>

namespace turbodbc { namespace bindings {

void for_buffer_size()
{
    boost::python::class_<turbodbc::rows>("Rows", boost::python::init<std::size_t>())
        .def_readwrite("rows_to_buffer", &turbodbc::rows::rows_to_buffer)
    ;
}

} }
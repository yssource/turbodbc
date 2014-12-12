/**
 *  @file dummy.cpp
 *  @date 05.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <boost/python.hpp>
#include <cpp_odbc/cpp_odbc.h>

#include <sqlext.h>

namespace bp = boost::python;

namespace pydbc {

	struct py_cursor {
		psapp::valid_ptr<cpp_odbc::statement> statement;

		void execute(std::string const & sql)
		{
			statement->execute(sql);
		}
	};

	struct py_connection {
		psapp::valid_ptr<cpp_odbc::connection> connection;

		void commit()
		{
			connection->commit();
		}

		py_cursor cursor()
		{
			return {psapp::to_valid(connection->make_statement())};
		}
	};

	py_connection connect(std::string const & data_source_name)
	{
		auto environment = cpp_odbc::make_environment();
		return {psapp::to_valid(environment->make_connection("dsn=" + data_source_name))};
	}

}

BOOST_PYTHON_MODULE(pydbc)
{
    bp::def("connect", pydbc::connect);

    bp::class_<pydbc::py_cursor>("Cursor", bp::no_init)
    		.def("execute", &pydbc::py_cursor::execute)
    	;

    bp::class_<pydbc::py_connection>("Connection", bp::no_init)
    		.def("commit", &pydbc::py_connection::commit)
    		.def("cursor", &pydbc::py_connection::cursor)
    	;
}

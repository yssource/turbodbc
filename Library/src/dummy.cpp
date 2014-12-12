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
#include <cpp_odbc/level1/unixodbc_backend.h>
#include <cpp_odbc/level2/level1_connector.h>
#include <cpp_odbc/raii_environment.h>
#include <cpp_odbc/connection.h>
#include <cpp_odbc/statement.h>

#include <psapp/valid_ptr.h>
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

	psapp::valid_ptr<cpp_odbc::environment> make_environment()
	{
		auto l1 = psapp::make_valid_ptr<cpp_odbc::level1::unixodbc_backend const>();
		auto l2 = psapp::make_valid_ptr<cpp_odbc::level2::level1_connector const>(l1);
		return psapp::make_valid_ptr<cpp_odbc::raii_environment>(l2);
	}

	py_connection connect(std::string const & data_source_name)
	{
		auto environment = make_environment();
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

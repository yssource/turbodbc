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
#include <cpp_odbc/raii_connection.h>
#include <cpp_odbc/raii_statement.h>

#include <psapp/valid_ptr.h>
#include <sqlext.h>

namespace bp = boost::python;
using namespace cpp_odbc;

namespace pydbc {

	struct py_cursor {
		psapp::valid_ptr<raii_environment> environment;
		psapp::valid_ptr<raii_connection> connection;
		psapp::valid_ptr<raii_statement> statement;

		void execute(std::string const & sql)
		{
			statement->execute(sql);
		}
	};

	struct py_connection {
		psapp::valid_ptr<raii_environment> environment;
		psapp::valid_ptr<raii_connection> connection;
		psapp::valid_ptr<raii_statement> statement;

		void commit()
		{
			auto api = environment->get_api();
			api->end_transaction(connection->get_handle(), SQL_COMMIT);
		}

		py_cursor cursor()
		{
			return {environment, connection, statement};
		}
	};

	py_connection connect(std::string const & data_source_name)
	{
		auto l1 = psapp::make_valid_ptr<level1::unixodbc_backend const>();
		auto l2 = psapp::make_valid_ptr<level2::level1_connector const>(l1);
		auto environment = psapp::make_valid_ptr<raii_environment>(l2);
		auto connection = psapp::make_valid_ptr<raii_connection>(l2, environment->get_handle(), "dsn=" + data_source_name);
		auto statement = psapp::make_valid_ptr<raii_statement>(l2, connection->get_handle());

		return {environment, connection, statement};
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

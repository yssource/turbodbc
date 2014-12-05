/**
 *  @file raii_connection.cpp
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-12-05 08:55:14 +0100 (Fr, 05 Dez 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21240 $
 *
 */

#include "cpp_odbc/raii_connection.h"

#include "cpp_odbc/raii_environment.h"
#include "cpp_odbc/level2/api.h"
#include "cpp_odbc/level2/handles.h"

#include <boost/format.hpp>

#include <mutex>

namespace {
	// this lock should be used whenever a connection is connected/disconnected
	static std::mutex create_destroy_mutex;

	// raii just for connection handle
	struct raii_handle {
		psapp::valid_ptr<cpp_odbc::level2::api const> api;
		cpp_odbc::level2::connection_handle handle;

		raii_handle(psapp::valid_ptr<cpp_odbc::level2::api const> api,
				cpp_odbc::level2::environment_handle const & environment) :
			api(std::move(api)),
			handle(api->allocate_connection_handle(environment))
		{
		}

		~raii_handle()
		{
			api->free_handle(handle);
		}
	};
}

namespace cpp_odbc {

struct raii_connection::intern {
	raii_handle handle;
	psapp::valid_ptr<cpp_odbc::level2::api const> api;

	intern(
			psapp::valid_ptr<cpp_odbc::level2::api const> api,
			cpp_odbc::level2::environment_handle const & environment,
			std::string const & connection_string
		) :
		handle(api, environment),
		api(std::move(api))
	{
		thread_safe_establish_connection(connection_string);
	}

	~intern()
	{
		thread_safe_disconnect();
	}

private:

	void thread_safe_establish_connection(std::string const & data_source_name)
	{
		std::lock_guard<std::mutex> guard(create_destroy_mutex);
		api->establish_connection(handle.handle, data_source_name);
	}

	void thread_safe_disconnect()
	{
		std::lock_guard<std::mutex> guard(create_destroy_mutex);
		api->disconnect(handle.handle);
	}
};


raii_connection::raii_connection(psapp::valid_ptr<cpp_odbc::level2::api const> api, cpp_odbc::level2::environment_handle const & environment, std::string const & connection_string) :
	impl_(std::move(api), environment, connection_string)
{
}


}

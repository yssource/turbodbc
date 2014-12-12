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

#include "cpp_odbc/raii_statement.h"
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
			psapp::valid_ptr<raii_environment const> environment,
			std::string const & connection_string
		) :
		handle(api, environment->get_handle()),
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


raii_connection::raii_connection(psapp::valid_ptr<cpp_odbc::level2::api const> api, psapp::valid_ptr<raii_environment const> environment, std::string const & connection_string) :
	impl_(std::move(api), environment, connection_string)
{
}

psapp::valid_ptr<level2::api const> raii_connection::get_api() const
{
	return impl_->api;
}

level2::connection_handle const & raii_connection::get_handle() const
{
	return impl_->handle.handle;
}

std::shared_ptr<statement> raii_connection::do_make_statement() const
{
	auto as_valid_raii_connection = psapp::to_valid(std::dynamic_pointer_cast<raii_connection const>(shared_from_this()));
	return std::make_shared<raii_statement>(as_valid_raii_connection);
}

void raii_connection::do_set_connection_attribute(SQLINTEGER attribute, long value) const
{
	impl_->api->set_connection_attribute(impl_->handle.handle, attribute, value);
}

void raii_connection::do_commit() const
{
	impl_->api->end_transaction(impl_->handle.handle, SQL_COMMIT);
}

void raii_connection::do_rollback() const
{
	impl_->api->end_transaction(impl_->handle.handle, SQL_ROLLBACK);
}

std::string raii_connection::do_get_string_info(SQLUSMALLINT info_type) const
{
	return impl_->api->get_string_connection_info(impl_->handle.handle, info_type);
}

}

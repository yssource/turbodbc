/**
 *  @file level2_dummy_api.cpp
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc_test/level2_dummy_api.h"

namespace cpp_odbc_test {

level2_dummy_api::level2_dummy_api() = default;
level2_dummy_api::~level2_dummy_api() = default;

cpp_odbc::level2::statement_handle level2_dummy_api::do_allocate_statement_handle(cpp_odbc::level2::connection_handle const &) const
{
	return {nullptr};
}

cpp_odbc::level2::connection_handle level2_dummy_api::do_allocate_connection_handle(cpp_odbc::level2::environment_handle const &) const
{
	return {nullptr};
}

cpp_odbc::level2::environment_handle level2_dummy_api::do_allocate_environment_handle() const
{
	return {nullptr};
}

void level2_dummy_api::do_free_handle(cpp_odbc::level2::statement_handle &) const
{
}

void level2_dummy_api::do_free_handle(cpp_odbc::level2::connection_handle &) const
{
}

void level2_dummy_api::do_free_handle(cpp_odbc::level2::environment_handle &) const
{
}


void level2_dummy_api::do_set_environment_attribute(cpp_odbc::level2::environment_handle const &, SQLINTEGER, long) const
{
}

void level2_dummy_api::do_set_connection_attribute(cpp_odbc::level2::connection_handle const &, SQLINTEGER, long) const
{
}

void level2_dummy_api::do_establish_connection(cpp_odbc::level2::connection_handle &, std::string const &) const
{
}

void level2_dummy_api::do_disconnect(cpp_odbc::level2::connection_handle &) const
{
}

void level2_dummy_api::do_end_transaction(cpp_odbc::level2::connection_handle const &, SQLSMALLINT) const
{
}

std::string level2_dummy_api::do_get_string_connection_info(cpp_odbc::level2::connection_handle const &, SQLUSMALLINT) const
{
	return "dummy";
}

void level2_dummy_api::do_execute_prepared_statement(cpp_odbc::level2::statement_handle const &) const
{
}

void level2_dummy_api::do_execute_statement(cpp_odbc::level2::statement_handle const &, std::string const &) const
{
}

void level2_dummy_api::do_fetch_scroll(cpp_odbc::level2::statement_handle const &, SQLSMALLINT, SQLLEN) const
{
}

void level2_dummy_api::do_free_statement(cpp_odbc::level2::statement_handle const &, SQLUSMALLINT) const
{
}

void level2_dummy_api::do_prepare_statement(cpp_odbc::level2::statement_handle const &, std::string const &) const
{
}


}

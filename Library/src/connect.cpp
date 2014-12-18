/**
 *  @file connect.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "pydbc/connect.h"
#include <cpp_odbc/make_environment.h>

pydbc::connection pydbc::connect(std::string const & data_source_name)
{
	auto environment = cpp_odbc::make_environment();
	return {environment->make_connection("dsn=" + data_source_name)};
}

/**
 *  @file make_environment.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "cpp_odbc/make_environment.h"

#include "cpp_odbc/level3/raii_environment.h"
#include "cpp_odbc/level1/unixodbc_backend.h"
#include "cpp_odbc/level2/level1_connector.h"


namespace cpp_odbc {

psapp::valid_ptr<environment> make_environment()
{
	auto level1_api = psapp::make_valid_ptr<level1::unixodbc_backend const>();
	auto level2_api = psapp::make_valid_ptr<level2::level1_connector const>(level1_api);
	return psapp::make_valid_ptr<level3::raii_environment>(level2_api);
}

}

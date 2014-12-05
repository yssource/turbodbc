#pragma once
/**
 *  @file raii_connection.h
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-12-05 08:55:14 +0100 (Fr, 05 Dez 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21240 $
 *
 */


#include "cpp_odbc/level2/handles.h"

#include "psapp/pattern/pimpl.h"
#include "psapp/valid_ptr_core.h"

#include <string>

namespace cpp_odbc { namespace level2 {
	class api;
} }

namespace cpp_odbc {

class raii_connection {
public:
	raii_connection(psapp::valid_ptr<cpp_odbc::level2::api const> api, cpp_odbc::level2::environment_handle const & environment, std::string const & connection_string);

private:
	struct intern;
	psapp::pattern::pimpl<raii_connection> impl_;
};

}

#pragma once
/**
 *  @file connect.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "pydbc/connection.h"
#include <string>

namespace pydbc {

py_connection connect(std::string const & data_source_name);

}

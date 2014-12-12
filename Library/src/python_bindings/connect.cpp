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

#include <pydbc/connect.h>

#include <boost/python/def.hpp>

namespace pydbc { namespace bindings {

void for_connect()
{
    boost::python::def("connect", pydbc::connect);
}

} }

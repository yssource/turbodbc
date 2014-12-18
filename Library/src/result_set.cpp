/**
 *  @file cursor.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/cursor.h>
#include <sqlext.h>
#include <stdexcept>

namespace pydbc {

std::size_t const cached_rows = 10;

result_set::result_set(std::size_t number_of_columns) :
	columns(number_of_columns, cpp_odbc::multi_value_buffer(sizeof(long), cached_rows))
{
}

}

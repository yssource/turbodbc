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

namespace pydbc {

void cursor::execute(std::string const & sql)
{
	statement->execute(sql);
}

std::vector<int> cursor::fetch_one()
{
	return {17};
}

}

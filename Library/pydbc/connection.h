#pragma once
/**
 *  @file connection.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include "pydbc/cursor.h"
#include <cpp_odbc/connection.h>
#include <memory>

namespace pydbc {

/**
 * @brief This class is provides basic functionality required by python's
 *        connection class as specified by the database API version 2.
 *        Additional wrapping may be required.
 */
class connection {
public:
	/**
	 * @brief Construct a new connection based on the given low-level connection
	 */
	connection(std::shared_ptr<cpp_odbc::connection> low_level_connection);

	/**
	 * @brief Commit all operations which have been performed since the last commit
	 *        or rollback
	 */
	void commit();

	/**
	 * @brief Create a new cursor object associated with this connection
	 */
	pydbc::cursor make_cursor();
private:
	std::shared_ptr<cpp_odbc::connection> connection_;
};

}

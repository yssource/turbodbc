#pragma once
/**
 *  @file cursor.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <cpp_odbc/statement.h>
#include <pydbc/result_set.h>
#include <memory>
#include <vector>

namespace pydbc {

/**
 * TODO: Cursor needs proper unit tests
 */
class cursor {
public:
	cursor(std::shared_ptr<cpp_odbc::statement const> statement);

	void execute(std::string const & sql);
	std::vector<nullable_field> fetch_one();
	long get_rowcount();

	std::shared_ptr<cpp_odbc::statement const> get_statement() const;

private:
	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::shared_ptr<result_set> result_;
};

}

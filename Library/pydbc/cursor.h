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
#include <pydbc/parameter.h>
#include <memory>
#include <vector>

namespace pydbc {

/**
 * TODO: Cursor needs proper unit tests
 */
class cursor {
public:
	cursor(std::shared_ptr<cpp_odbc::statement const> statement);

	void prepare(std::string const & sql);
	void execute();
	void bind_parameters();
	void add_parameter_set(std::vector<nullable_field> const & parameter_set);

	std::vector<nullable_field> fetch_one();
	long get_rowcount();

	std::shared_ptr<cpp_odbc::statement const> get_statement() const;

	~cursor();

private:
	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::vector<std::shared_ptr<pydbc::parameter>> parameters_;
	std::shared_ptr<result_set> result_;
	std::size_t current_parameter_set_;
};

}

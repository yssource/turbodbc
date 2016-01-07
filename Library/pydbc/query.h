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
 * TODO: Query needs proper unit tests
 */
class query {
public:
	query(std::shared_ptr<cpp_odbc::statement const> statement);

	void execute();
	void bind_parameters();
	void add_parameter_set(std::vector<nullable_field> const & parameter_set);

	std::vector<nullable_field> fetch_one();
	long get_row_count();

	~query();

private:
	void check_parameter_set(std::vector<nullable_field> const & parameter_set) const;

	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::vector<std::shared_ptr<pydbc::parameter>> parameters_;
	std::shared_ptr<result_set> result_;
	std::size_t current_parameter_set_;
};

}

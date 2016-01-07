#pragma once

#include <cpp_odbc/connection.h>
#include <pydbc/query.h>
#include <memory>


namespace pydbc {

/**
 * TODO: Cursor needs proper unit tests
 */
class cursor {
public:
	cursor(std::shared_ptr<cpp_odbc::connection const> connection);

	void prepare(std::string const & sql);
	void execute();
	void add_parameter_set(std::vector<nullable_field> const & parameter_set);

	std::vector<nullable_field> fetch_one();
	long get_row_count();

	std::shared_ptr<cpp_odbc::connection const> get_connection() const;

	~cursor();

private:
	std::shared_ptr<cpp_odbc::connection const> connection_;
	std::shared_ptr<pydbc::query> query_;
};

}

#include <pydbc/column_types.h>
#include <sqlext.h>

namespace pydbc {

long_column::long_column(cpp_odbc::statement const & statement, std::size_t one_based_index) :
		buffer_(sizeof(long), cached_rows)
{
	statement.bind_column(one_based_index, SQL_C_SBIGINT, buffer_);
}

nullable_field long_column::do_get_field() const
{
	auto const element = buffer_[0];
	if (element.indicator == SQL_NULL_DATA) {
		return {};
	} else {
		auto value_ptr = reinterpret_cast<long const *>(element.data_pointer);
		return field{*value_ptr};
	}
}


namespace {
	std::size_t const maximum_string_size = 1024;
}

string_column::string_column(cpp_odbc::statement const & statement, std::size_t one_based_index) :
		buffer_(maximum_string_size, cached_rows)
{
	statement.bind_column(one_based_index, SQL_CHAR, buffer_);
}

nullable_field string_column::do_get_field() const
{
	auto const element = buffer_[0];
	if (element.indicator == SQL_NULL_DATA) {
		return {};
	} else {
		return field{std::string(buffer_[0].data_pointer)}; // unixodbc stores null-terminated strings
	}
}

}


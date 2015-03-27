#pragma once

#include <pydbc/description.h>

namespace pydbc {

/**
 * @brief Represents a description to bind a buffer holding integer values
 */
class string_description : public description {
public:
	string_description(std::size_t maximum_length);
	~string_description();
private:
	std::size_t do_element_size() const final;
	SQLSMALLINT do_column_type() const final;
	SQLSMALLINT do_column_sql_type() const final;
	field do_make_field(char const * data_pointer) const final;

	std::size_t maximum_length_;
};

}

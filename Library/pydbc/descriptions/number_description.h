#pragma once

#include <pydbc/description.h>

namespace pydbc {

/**
 * @brief Represents a description to bind a buffer holding number values
 *        (a.k.a. 128 bit integer with decimal point)
 */
class number_description : public description {
public:
	number_description();
	~number_description();
private:
	std::size_t do_element_size() const final;
	SQLSMALLINT do_column_c_type() const final;
	SQLSMALLINT do_column_sql_type() const final;
	field do_make_field(char const * data_pointer) const final;
};


}

#pragma once

#include <pydbc/description.h>

namespace pydbc {

/**
 * @brief Represents a description to bind a buffer holding boolean values
 */
class boolean_description : public description {
public:
	boolean_description();
	~boolean_description();
private:
	std::size_t do_element_size() const final;
	SQLSMALLINT do_column_type() const final;
	field do_make_field(char const * data_pointer) const final;
};

}

#pragma once

#include "pydbc/field.h"
#include <sqltypes.h>

namespace pydbc {

/**
 * @brief Represents all information to bind a buffer to a column and
 *        how to interpret values stored in there
 */
class description {
public:
	std::size_t element_size() const;
	SQLSMALLINT column_type() const;
	field make_field(char const * data_pointer) const;

	description (description const &) = delete;
	description & operator=(description const &) = delete;

	virtual ~description();
protected:
	description();
private:
	virtual std::size_t do_element_size() const = 0;
	virtual SQLSMALLINT do_column_type() const = 0;
	virtual field do_make_field(char const * data_pointer) const = 0;
};

}

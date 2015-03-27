#pragma once

#include "pydbc/field.h"
#include <sqltypes.h>

namespace pydbc {

/**
 * @brief Represents all information to bind a buffer to a column or parameter and
 *        how to store/load values
 */
class description {
public:
	/**
	 * @brief Returns the size of an element in a buffer
	 */
	std::size_t element_size() const;

	/**
	 * @brief Returns the type code for the associated C data type
	 */
	SQLSMALLINT column_type() const;

	/**
	 * @brief Returns the type code for the associated SQL column data type
	 */
	SQLSMALLINT column_sql_type() const;

	/**
	 * @brief Return a field based on the bytes pointed to by data_pointer
	 */
	field make_field(char const * data_pointer) const;

	description (description const &) = delete;
	description & operator=(description const &) = delete;

	virtual ~description();
protected:
	description();
private:
	virtual std::size_t do_element_size() const = 0;
	virtual SQLSMALLINT do_column_type() const = 0;
	virtual SQLSMALLINT do_column_sql_type() const = 0;
	virtual field do_make_field(char const * data_pointer) const = 0;
};

}

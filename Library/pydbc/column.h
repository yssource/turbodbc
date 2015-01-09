#pragma once

#include <pydbc/field.h>

namespace pydbc {

/**
 * @brief This interface represents a single column of a result set.
 */
struct column {
	/**
	 * @brief Retrieve the field of the current result set row associated with this column
	 */
	field get_field() const;

	column(column const &) = delete;
	column & operator=(column const &) = delete;

	virtual ~column();
protected:
	column();
private:
	virtual field do_get_field() const = 0;
};


}

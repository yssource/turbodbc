#pragma once

#include <turbodbc/field.h>
#include <turbodbc/parameter.h>

namespace turbodbc {

/**
 * @brief Check whether a parameter can hold the given value
 */
bool parameter_is_suitable_for(parameter const &param, field const &value);

/**
 * @brief Set the destination's buffer element to a value corresponding to the
 *        input field
 * @param value The input value
 * @param destination The target which value shall be changed
 */
//void set_field(field const & value, mutable_buffer_element & destination);

}
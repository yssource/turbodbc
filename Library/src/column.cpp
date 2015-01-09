#include <pydbc/column.h>

namespace pydbc {

column::column() = default;

column::~column() = default;

field column::get_field() const
{
	return do_get_field();
}

}

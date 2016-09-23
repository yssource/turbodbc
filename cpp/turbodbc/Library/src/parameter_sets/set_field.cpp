#include <turbodbc/parameter_sets/set_field.h>

#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>


namespace turbodbc {

namespace {

	std::size_t const size_not_important = 0;

	class is_suitable_for : public boost::static_visitor<bool> {
	public:
		is_suitable_for(parameter const & param) :
				parameter_(param)
		{}

		bool operator()(bool const &) const {
			return parameter_.is_suitable_for(type_code::boolean, size_not_important);
		}

		bool operator()(long const &) const {
			return parameter_.is_suitable_for(type_code::integer, size_not_important);
		}

		bool operator()(double const &) const {
			return parameter_.is_suitable_for(type_code::floating_point, size_not_important);
		}

		bool operator()(boost::posix_time::ptime const &) const {
			return parameter_.is_suitable_for(type_code::timestamp, size_not_important);
		}

		bool operator()(boost::gregorian::date const &) const {
			return parameter_.is_suitable_for(type_code::date, size_not_important);
		}

		bool operator()(std::string const & value) const {
			return parameter_.is_suitable_for(type_code::string, value.size() + 1);
		}

	private:
		parameter const & parameter_;
	};

}


bool parameter_is_suitable_for(parameter const & param, field const & value)
{
	return boost::apply_visitor(is_suitable_for(param), value);
}

}
#include <turbodbc/make_description.h>

#include <cppunit/extensions/HelperMacros.h>

#include <turbodbc/descriptions.h>

#include <sqlext.h>
#include <sstream>
#include <stdexcept>


class make_description_of_value_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( make_description_of_value_test );

	CPPUNIT_TEST( from_integer );
	CPPUNIT_TEST( from_double );
	CPPUNIT_TEST( from_bool );
	CPPUNIT_TEST( from_date );
	CPPUNIT_TEST( from_ptime );
	CPPUNIT_TEST( from_string_provides_minimum_length );
	CPPUNIT_TEST( from_string_provides_extra_space_for_large_strings );

CPPUNIT_TEST_SUITE_END();

public:

	void from_integer();
	void from_double();
	void from_bool();
	void from_date();
	void from_ptime();
	void from_string_provides_minimum_length();
	void from_string_provides_extra_space_for_large_strings();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( make_description_of_value_test );

using turbodbc::make_description;
using turbodbc::field;


void make_description_of_value_test::from_integer()
{
	field const value = 42l;
	auto description = make_description(value);
	CPPUNIT_ASSERT( dynamic_cast<turbodbc::integer_description const *>(description.get()) );
}

void make_description_of_value_test::from_double()
{
	field const value = 3.14;
	auto description = make_description(value);
	CPPUNIT_ASSERT( dynamic_cast<turbodbc::floating_point_description const *>(description.get()) );
}

void make_description_of_value_test::from_bool()
{
	field const value = true;
	auto description = make_description(value);
	CPPUNIT_ASSERT( dynamic_cast<turbodbc::boolean_description const *>(description.get()) );
}

void make_description_of_value_test::from_date()
{
	field const value = boost::gregorian::date(2016, 1, 7);
	auto description = make_description(value);
	CPPUNIT_ASSERT( dynamic_cast<turbodbc::date_description const *>(description.get()) );
}

void make_description_of_value_test::from_ptime()
{
	field const value = boost::posix_time::ptime({2016, 1, 7}, {1, 2, 3, 123456});
	auto description = make_description(value);
	CPPUNIT_ASSERT( dynamic_cast<turbodbc::timestamp_description const *>(description.get()) );
}

void make_description_of_value_test::from_string_provides_minimum_length()
{
	std::string small_string("hi");
	field const value(small_string);
	auto description = make_description(value);
	auto as_string_description = dynamic_cast<turbodbc::string_description const *>(description.get());
	CPPUNIT_ASSERT( as_string_description != nullptr );

	std::size_t const minimum_length = 10;
	CPPUNIT_ASSERT_EQUAL(as_string_description->element_size(), minimum_length + 1);
}

void make_description_of_value_test::from_string_provides_extra_space_for_large_strings()
{
	std::string large_string("this is a relatively large string");
	field const value(large_string);
	auto description = make_description(value);
	auto as_string_description = dynamic_cast<turbodbc::string_description const *>(description.get());
	CPPUNIT_ASSERT( as_string_description != nullptr );

	CPPUNIT_ASSERT(as_string_description->element_size() > (large_string.size() + 1));
}

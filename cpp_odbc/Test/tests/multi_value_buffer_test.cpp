/**
 *  @file multi_value_buffer_test.cpp
 *  @date 11.04.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:57:29 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21205 $
 *
 */


#include "cpp_odbc/multi_value_buffer.h"

#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/extensions/assert_equal_with_different_types.h"

#include <cstring>
#include <stdexcept>

class multi_value_buffer_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( multi_value_buffer_test );

	CPPUNIT_TEST( constructor_enforces_positive_parameters );
	CPPUNIT_TEST( capacity_per_element );
	CPPUNIT_TEST( data_pointer );
	CPPUNIT_TEST( indicator_pointer );
	CPPUNIT_TEST( mutable_element_access );
	CPPUNIT_TEST( const_element_access );

CPPUNIT_TEST_SUITE_END();

public:

	void constructor_enforces_positive_parameters();
	void capacity_per_element();
	void data_pointer();
	void indicator_pointer();
	void mutable_element_access();
	void const_element_access();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( multi_value_buffer_test );

using cpp_odbc::multi_value_buffer;

void multi_value_buffer_test::constructor_enforces_positive_parameters()
{
	CPPUNIT_ASSERT_THROW( multi_value_buffer(0, 1), std::logic_error );
	CPPUNIT_ASSERT_THROW( multi_value_buffer(1, 0), std::logic_error );
	CPPUNIT_ASSERT_NO_THROW( multi_value_buffer(1, 1) );
}

void multi_value_buffer_test::capacity_per_element()
{
	std::size_t const element_size = 100;
	multi_value_buffer buffer(element_size, 42);

	CPPUNIT_ASSERT_EQUAL( element_size, buffer.capacity_per_element() );
}

void multi_value_buffer_test::data_pointer()
{
	std::size_t const element_size = 100;
	std::size_t const number_of_elements = 42;
	multi_value_buffer buffer(element_size, number_of_elements);

	// write something to have some protection against segmentation faults
	std::memset( buffer.data_pointer(), 0xff, element_size * number_of_elements );
}

void multi_value_buffer_test::indicator_pointer()
{
	std::size_t const number_of_elements = 42;
	multi_value_buffer buffer(3, number_of_elements);

	// write something to have some protection against segmentation faults
	buffer.indicator_pointer()[number_of_elements - 1] = 17;
}

void multi_value_buffer_test::mutable_element_access()
{
	std::size_t const element_size = 3;
	std::size_t const number_of_elements = 2;
	multi_value_buffer buffer(element_size, number_of_elements);

	std::strcpy( buffer[0].data_pointer, "abc" );
	std::strcpy( buffer[1].data_pointer, "def" );
	CPPUNIT_ASSERT_EQUAL( 0, std::memcmp(buffer.data_pointer(), "abcdef", 6));

	long const expected_indicator = 42;
	buffer[1].indicator = expected_indicator;
	CPPUNIT_ASSERT_EQUAL( expected_indicator, buffer.indicator_pointer()[1]);
}

void multi_value_buffer_test::const_element_access()
{
	std::size_t const element_size = 3;
	std::size_t const number_of_elements = 2;
	multi_value_buffer buffer(element_size, number_of_elements);

	std::string const data = "abcdef";
	long const expected_indicator = 17;

	std::strcpy( buffer.data_pointer(), data.c_str() );
	buffer.indicator_pointer()[1] = expected_indicator;

	auto const & const_buffer = buffer;

	CPPUNIT_ASSERT_EQUAL( data[element_size], *const_buffer[1].data_pointer );
	CPPUNIT_ASSERT_EQUAL( expected_indicator, const_buffer[1].indicator );
}

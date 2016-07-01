#pragma once

#include <turbodbc/result_sets/bound_result_set.h>
#include <turbodbc/column.h>

#include <cpp_odbc/statement.h>
#include <memory>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>


namespace turbodbc { namespace result_sets {

namespace detail {

	/**
	 * @brief Implement a very basic thread-safe message queue
	 */
	class message_queue {
	public:
		message_queue();
		~message_queue();
		void push(std::size_t value);
		std::size_t pull();
	private:
		std::mutex mutex_;
		std::condition_variable condition_;
		std::queue<std::size_t> messages_;
	};

}

/**
 * @brief This class implements result_set by double buffering real ODBC
 *        result sets. This means that while one buffer is filled by the database,
 *        users retrieve values from a previously filled buffer.
 */
class double_buffered_result_set : public turbodbc::result_sets::result_set {
public:
	/**
	 * @brief Prepare and bind buffers suitable of holding buffered_rows to
	 *        the given statement.
	 */
	double_buffered_result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows);
	virtual ~double_buffered_result_set();

private:
	std::size_t do_fetch_next_batch() final;
	std::vector<column_info> do_get_column_info() const final;
	std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> do_get_buffers() const final;

	std::shared_ptr<cpp_odbc::statement const> statement_;
	std::array<bound_result_set, 2> batches_;
	std::size_t active_reading_batch_;
	detail::message_queue read_requests_;
	detail::message_queue read_responses_;
	std::thread reader_;
};


} }

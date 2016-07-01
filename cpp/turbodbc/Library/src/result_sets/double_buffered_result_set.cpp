#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <turbodbc/make_description.h>

#include <sqlext.h>

#include <future>

namespace turbodbc { namespace result_sets {

namespace detail {
	message_queue::message_queue() = default;
	message_queue::~message_queue() = default;


	void message_queue::push(std::size_t value)
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			messages_.push(value);
		}
		condition_.notify_one();
	}

	std::size_t message_queue::pull()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		condition_.wait(lock, [&](){return not messages_.empty();});
		auto const value = messages_.front();
		messages_.pop();
		return value;
	}
}


namespace {

	void reader_thread(detail::message_queue & read_requests,
	                   detail::message_queue & read_responses,
	                   std::array<bound_result_set, 2> & batches)
	{
		std::size_t batch_id = 0;
		do {
			batch_id = read_requests.pull();
			if (batch_id != 2) {
				batches[batch_id].rebind();
				auto const n_rows = batches[batch_id].fetch_next_batch();
				read_responses.push(n_rows);
			}
		} while (batch_id != 2);
	}

	std::size_t rows_per_single_buffer(std::size_t buffered_rows)
	{
		return (buffered_rows / 2 + buffered_rows % 2);
	}

}


double_buffered_result_set::double_buffered_result_set(std::shared_ptr<cpp_odbc::statement const> statement, std::size_t buffered_rows) :
	statement_(statement),
	batches_{{bound_result_set(statement_, rows_per_single_buffer(buffered_rows)),
	          bound_result_set(statement_, rows_per_single_buffer(buffered_rows))}},
	active_reading_batch_(0),
	reader_(reader_thread,
	        std::ref(read_requests_),
	        std::ref(read_responses_),
	        std::ref(batches_))
{
	read_requests_.push(active_reading_batch_);
	active_reading_batch_ = 1;
}

double_buffered_result_set::~double_buffered_result_set()
{
	read_requests_.push(2);
	reader_.join();
}


std::size_t double_buffered_result_set::do_fetch_next_batch()
{
	read_requests_.push(active_reading_batch_);
	active_reading_batch_ = (active_reading_batch_ == 0) ? 1 : 0;
	return read_responses_.pull();
}


std::vector<column_info> double_buffered_result_set::do_get_column_info() const
{
	return batches_[active_reading_batch_].get_column_info();
}


std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> double_buffered_result_set::do_get_buffers() const
{
	return batches_[active_reading_batch_].get_buffers();
}

} }

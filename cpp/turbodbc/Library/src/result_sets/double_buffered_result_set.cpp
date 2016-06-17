#include <turbodbc/result_sets/double_buffered_result_set.h>

#include <turbodbc/make_description.h>

#include <sqlext.h>

#include <future>

namespace turbodbc { namespace result_sets {


namespace {

	void reader_thread(std::condition_variable & message_condition,
	                   std::mutex & message_mutex,
	                   std::queue<std::size_t> & messages,
	                   std::condition_variable & rows_condition,
	                   std::mutex & rows_mutex,
	                   std::queue<std::size_t> & rows,
	                   std::array<bound_result_set, 2> & batches)
	{
		std::unique_lock<std::mutex> lock(message_mutex);

		while (true) {
			message_condition.wait(lock, [&messages](){return not messages.empty();});
			std::size_t const batch_id = messages.front();
			messages.pop();
			if (batch_id != 2) {
				batches[batch_id].rebind();
				auto const n_rows = batches[batch_id].fetch_next_batch();
				std::lock_guard<std::mutex> lock(rows_mutex);
				rows.push(n_rows);
				rows_condition.notify_one();
			} else {
				break;
			}
		}
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
	        std::ref(message_condition_),
	        std::ref(message_mutex_),
	        std::ref(messages_),
	        std::ref(rows_condition_),
	        std::ref(rows_mutex_),
	        std::ref(rows_),
	        std::ref(batches_))
{
	trigger_next_fetch();
	active_reading_batch_ = 1;
}

double_buffered_result_set::~double_buffered_result_set()
{
	{
		std::lock_guard<std::mutex> lock(message_mutex_);
		messages_.push(2);
	}
	message_condition_.notify_one();
	reader_.join();
}


std::size_t double_buffered_result_set::do_fetch_next_batch()
{
	auto const new_active = (active_reading_batch_ == 0) ? 1 : 0;

	trigger_next_fetch();
	active_reading_batch_ = new_active;

	std::unique_lock<std::mutex> lock(rows_mutex_);
	rows_condition_.wait(lock, [&](){return not rows_.empty();});
	auto n_rows = rows_.front();
	rows_.pop();

	return n_rows;
}


std::vector<column_info> double_buffered_result_set::do_get_column_info() const
{
	return batches_[active_reading_batch_].get_column_info();
}


std::vector<std::reference_wrapper<cpp_odbc::multi_value_buffer const>> double_buffered_result_set::do_get_buffers() const
{
	return batches_[active_reading_batch_].get_buffers();
}

void double_buffered_result_set::trigger_next_fetch()
{
	{
		std::lock_guard<std::mutex> lock(message_mutex_);
		messages_.push(active_reading_batch_);
	}
	message_condition_.notify_one();
}

} }

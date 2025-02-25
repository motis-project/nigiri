#include "nigiri/loader/gtfs/fares/timeframes.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_time.h"

namespace nigiri::loader::gtfs::fares {

std::vector<timeframe> read_timeframes(std::string_view file_content) {
  struct timeframe_record {
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_id")> timeframe_id_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_name")> timeframe_name_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_start_time")>
        timeframe_start_time_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_end_time")> timeframe_end_time_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_duration")> timeframe_duration_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_disable_after_purchase")>
        timeframe_disable_after_purchase_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Timeframes")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                              progress_tracker->update_fn())} |
         utl::csv<timeframe_record>() |
         utl::transform([](timeframe_record const& r) {
           auto tf = timeframe{};
           tf.timeframe_id_ = r.timeframe_id_->view();

           // Handle optional fields
           if (!r.timeframe_name_->trim().empty()) {
             tf.timeframe_name_ = r.timeframe_name_->view();
           }

           if (!r.timeframe_start_time_->trim().empty()) {
             tf.timeframe_start_time_ =
                 nigiri::loader::gtfs::hhmm_to_min(*r.timeframe_start_time_);
           }

           if (!r.timeframe_end_time_->trim().empty()) {
             tf.timeframe_end_time_ =
                 nigiri::loader::gtfs::hhmm_to_min(*r.timeframe_end_time_);
           }

           if (!r.timeframe_duration_->trim().empty()) {
             tf.timeframe_duration_ =
                 utl::parse<unsigned>(*r.timeframe_duration_);
           }

           if (!r.timeframe_disable_after_purchase_->trim().empty()) {
             tf.timeframe_disable_after_purchase_ =
                 utl::parse<unsigned>(*r.timeframe_disable_after_purchase_) ==
                 1U;
           }

           return tf;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares
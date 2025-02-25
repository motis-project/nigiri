#include "nigiri/loader/gtfs/fares/fare_media.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_medium> read_fare_media(std::string_view file_content) {
  struct fare_media_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_id")> fare_media_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_name")> fare_media_name_;
    utl::csv_col<unsigned, UTL_NAME("fare_media_type")> fare_media_type_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_restrictions")> fare_media_restrictions_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Media")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<fare_media_record>() |
         utl::transform([](fare_media_record const& r) {
           auto medium = fare_medium{};
           medium.fare_media_id_ = r.fare_media_id_->view();
           
           // Handle media type enum
           switch (*r.fare_media_type_) {
             case 0U: medium.media_type_ = fare_media_type::kPhysical; break;
             case 1U: medium.media_type_ = fare_media_type::kVirtual; break;
             default: medium.media_type_ = fare_media_type::kPhysical; break;
           }
           
           // Handle optional fields
           if (!r.fare_media_name_->trim().empty()) {
             medium.fare_media_name_ = r.fare_media_name_->view();
           }
           
           // Parse restrictions if present
           if (!r.fare_media_restrictions_->trim().empty()) {
             auto restrictions_value = utl::parse<unsigned>(*r.fare_media_restrictions_);
             switch (restrictions_value) {
               case 0U: medium.restrictions_ = fare_media_restriction::kNone; break;
               case 1U: medium.restrictions_ = fare_media_restriction::kReserveFirstUse; break;
               case 2U: medium.restrictions_ = fare_media_restriction::kReserveBeforeUse; break;
               default: medium.restrictions_ = fare_media_restriction::kNone; break;
             }
           }
           
           return medium;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares
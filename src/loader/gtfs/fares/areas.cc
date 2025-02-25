#include "nigiri/loader/gtfs/fares/areas.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<area> read_areas(std::string_view file_content) {
  struct area_record {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("area_name")> area_name_;
    utl::csv_col<utl::cstr, UTL_NAME("area_type")> area_type_;
    utl::csv_col<utl::cstr, UTL_NAME("geometry_id")> geometry_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Areas")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<area_record>() |
         utl::transform([](area_record const& r) {
           auto a = area{};
           a.area_id_ = r.area_id_->view();
           
           // Handle optional fields
           if (!r.area_name_->trim().empty()) {
             a.area_name_ = r.area_name_->view();
           }
           
           if (!r.area_type_->trim().empty()) {
             auto type_value = utl::parse<unsigned>(*r.area_type_);
             switch (type_value) {
               case 0U: a.area_type_ = area_type::kStop; break;
               case 1U: a.area_type_ = area_type::kZone; break;
               case 2U: a.area_type_ = area_type::kRoute; break;
               default: a.area_type_ = area_type::kStop; break;
             }
           }
           
           if (!r.geometry_id_->trim().empty()) {
             a.geometry_id_ = r.geometry_id_->view();
           }
           
           return a;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares
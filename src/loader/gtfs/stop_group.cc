#include "nigiri/loader/gtfs/stop_group.h"

#include <cmath>
#include <algorithm>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

void add_stop_groups(timetable& tt,
                     std::string_view stop_group_elements_file_content,
                     stops_map_t const& stops) {
  struct stop_group_element_record {
    utl::csv_col<utl::cstr, UTL_NAME("stop_group_id")> stop_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  if (stop_group_elements_file_content.empty()) {
    return;
  }

  utl::line_range{utl::make_buf_reader(stop_group_elements_file_content)}  //
      | utl::csv<stop_group_element_record>()  //
      | utl::for_each([&](stop_group_element_record const& r) {
          auto const stop_group_id = r.stop_group_id_->trim().view();
          auto const stop_id = r.stop_id_->trim().view();

          auto const meta_it = stops.find(stop_group_id);
          if (meta_it == end(stops)) {
            log(log_lvl::error, "loader.gtfs.stop_groups",
                "stop_group_id={:?} not found", stop_group_id);
            return;
          }

          auto const stop_it = stops.find(stop_id);
          if (stop_it == end(stops)) {
            log(log_lvl::error, "loader.gtfs.stop_groups",
                "stop_id={:?} not found", stop_id);
            return;
          }

          auto const meta_l = meta_it->second;
          if (tt.locations_.coordinates_[meta_l] != geo::latlng{0, 0}) {
            log(log_lvl::error, "loader.gtfs.stop_groups",
                "stop_group_id=\"{}\" is not at (0,0)", stop_group_id);
            return;
          }

          auto const stop_l = stop_it->second;
          tt.locations_.equivalences_[meta_l].emplace_back(stop_l);
          for (auto const child1 : tt.locations_.children_[stop_l]) {
            tt.locations_.equivalences_[meta_l].emplace_back(child1);
            for (auto const child2 : tt.locations_.children_[child1]) {
              tt.locations_.equivalences_[meta_l].emplace_back(child2);
            }
          }
        });
}

}  // namespace nigiri::loader::gtfs

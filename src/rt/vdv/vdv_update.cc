#include "nigiri/rt/vdv/vdv_update.h"

#include <string>
#include <string_view>
#include <unordered_set>

#include "pugixml.hpp"
#include "utl/verify.h"

#include "nigiri/logging.h"
#include "nigiri/rt/vdv/vdv_run.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

constexpr auto const kAllowedError = 5_minutes;

struct time_of_day_interval {
  bool contains(std::uint16_t t) const {
    return start_ <= end_ ? (start_ <= t && t <= end_)
                          : (start_ <= t || t <= end_);
  }

  std::uint16_t start_;
  std::uint16_t end_;
};

std::optional<location_idx_t> match_location(timetable const& tt,
                                             std::string_view vdv_stop_id) {
  auto loc_match = std::optional<location_idx_t>{};
  for (auto l = 0U; l != tt.n_locations(); ++l) {
    auto const loc_idx = location_idx_t{l};
    if (std::string_view{begin(tt.locations_.ids_[loc_idx]),
                         end(tt.locations_.ids_[loc_idx])} == vdv_stop_id) {
      loc_match = loc_idx;
      break;
    }
  }
  return loc_match;
}

template <event_type ET>
void match_time(timetable const& tt,
                location_idx_t const loc_idx,
                unixtime_t const time,
                std::unordered_set<transport_idx_t>& matches) {
  auto const unixtime_intvl = interval<unixtime_t>{
      time - kAllowedError, time + kAllowedError + 1_minutes};
  auto const time_of_day_intvl = time_of_day_interval{
      static_cast<uint16_t>(unixtime_intvl.from_.time_since_epoch().count() %
                            1440),
      static_cast<std::uint16_t>(unixtime_intvl.to_.time_since_epoch().count() %
                                 1440)};

  for (auto const route_idx : tt.location_routes_[loc_idx]) {
    auto const loc_seq = tt.route_location_seq_[route_idx];
    for (auto stop_idx = 0U + (ET == event_type::kArr ? 1U : 0U);
         stop_idx != loc_seq.size() - (ET == event_type::kDep ? 1U : 0U);
         ++stop_idx) {
      auto const stp = stop{loc_seq[stop_idx]};
      if (stp.location_idx() != loc_idx) {
        continue;
      }
      auto const event_times_at_stop = tt.event_times_at_stop(
          route_idx, static_cast<stop_idx_t>(stop_idx), ET);
      // iterate span elements, index of elements tells you that the n-th
      // indexed transport of the route is the one whose times you are checking
      // right now
    }
  }
}

std::unordered_set<transport_idx_t> match_stops(timetable const& tt,
                                                rt_timetable& rtt,
                                                source_idx_t const src,
                                                vdv_run const& r) {
  // make these static to reduce number of allocations?
  auto global_matches = std::unordered_set<transport_idx_t>{};
  auto local_matches = std::unordered_set<transport_idx_t>{};
  for (auto& vdv_stop : r.stops_) {

    auto const loc_idx = match_location(tt, vdv_stop.stop_id_);
    if (!loc_idx.has_value()) {
      log(log_lvl::error, "vdv_update.match_stops",
          "could not match stop {} to a location in the timetable",
          vdv_stop.stop_id_);
      continue;
    }

    if (vdv_stop.t_dep_.has_value()) {
      match_time<event_type::kDep>(tt, loc_idx.value(), vdv_stop.t_dep_.value(),
                                   local_matches);
    } else if (vdv_stop.t_arr_.has_value()) {
      match_time<event_type::kArr>(tt, loc_idx.value(), vdv_stop.t_arr_.value(),
                                   local_matches);
    }

    if (global_matches.empty()) {
      std::swap(global_matches, local_matches);
    } else {
      std::erase_if(global_matches,
                    [&](auto const& t) { return !local_matches.contains(t); });
      local_matches.clear();
    }

    if (global_matches.size() == 1) {
      break;
    }
  }

  return global_matches;
}

void process_run(timetable const& tt,
                 rt_timetable& rtt,
                 source_idx_t const src,
                 pugi::xml_node const& run_node) {
  auto const vdv_run = parse_run(run_node);
  auto transport_matches = match_stops(tt, rtt, src, vdv_run);
}

void vdv_update(timetable const& tt,
                rt_timetable& rtt,
                source_idx_t const src,
                std::string const& vdv_msg) {
  auto doc = pugi::xml_document{};
  auto result = doc.load_string(vdv_msg.c_str());
  utl::verify(result, "XML [{}] parsed with errors: {}\n", vdv_msg,
              result.description());
  utl::verify(
      std::string_view{doc.first_child().name()} == "DatenAbrufenAntwort",
      "Invalid message type {} for vdv update", doc.first_child().name());

  auto const runs_xpath =
      doc.select_nodes("DatenAbrufenAntwort/AUSNachricht/IstFahrt");
  for (auto const& run_xpath : runs_xpath) {
    process_run(tt, rtt, src, run_xpath.node());
  }
}

}  // namespace nigiri::rt
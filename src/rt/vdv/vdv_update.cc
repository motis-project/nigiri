#include "nigiri/rt/vdv/vdv_update.h"

#include <string>
#include <string_view>
#include <unordered_set>

#include "pugixml.hpp"
#include "utl/verify.h"

#include "nigiri/logging.h"

namespace nigiri::rt {

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
  auto const [base_day, base_mam] = tt.day_idx_mam(time);
  auto const time_of_day_intvl = interval{base_mam.count() - kAllowedError,
                                          base_mam.count() + kAllowedError + 1};

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
      for (auto i = 0U; i != event_times_at_stop.size(); ++i) {

        auto const normalize_event_time =
            [&time_of_day_intvl](auto const event_time) -> std::int16_t {
          if (time_of_day_intvl.to_ < event_time) {
            return event_time - 1440;
          } else if (event_time < time_of_day_intvl.from_) {
            return event_time + 1440;
          } else {
            return event_time;
          }
        };

        auto const normalized_event_time =
            normalize_event_time(event_times_at_stop[i].mam());
        if (!time_of_day_intvl.contains(normalized_event_time)) {
          continue;
        }
        auto const midnight_shift = normalized_event_time < 0      ? -1
                                    : normalized_event_time < 1440 ? 0
                                                                   : 1;
        auto const transport_day = day_idx_t{
            base_day.v_ - event_times_at_stop[i].days() + midnight_shift};
        auto const transport_idx = tt.route_transport_ranges_[route_idx][i];
        if (tt.bitfields_[tt.transport_traffic_days_[transport_idx]].test(
                transport_day.v_)) {
          matches.insert(transport_idx);
        }
      }
    }
  }
}

std::unordered_set<transport_idx_t> match_stops(timetable const& tt,
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
                 [[maybe_unused]] rt_timetable& rtt,
                 pugi::xml_node const& run_node) {
  auto const vdv_run = parse_run(run_node);
  auto transport_matches = match_stops(tt, vdv_run);
  if (transport_matches.size() > 1) {
    // try to exclude based on trip name, operator, etc.
  }
}

void vdv_update(timetable const& tt,
                rt_timetable& rtt,
                [[maybe_unused]] source_idx_t const src,
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
    process_run(tt, rtt, run_xpath.node());
  }
}

}  // namespace nigiri::rt
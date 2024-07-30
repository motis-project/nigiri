#include "nigiri/rt/vdv/vdv_update.h"

#include <sstream>
#include <string>
#include <string_view>

#include "pugixml.hpp"

#include "utl/enumerate.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt::vdv {

unixtime_t parse_time(std::string const& str) {
  unixtime_t parsed;
  auto ss = std::stringstream{str};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

std::optional<unixtime_t> get_opt_time(pugi::xml_node const& node,
                                       char const* str) {
  auto const xpath = node.select_node(str);
  return xpath != nullptr
             ? std::optional{parse_time(xpath.node().child_value())}
             : std::nullopt;
}

std::optional<bool> get_opt_bool(
    pugi::xml_node const& node,
    char const* key,
    std::optional<bool> default_to = std::nullopt) {
  auto const xpath = node.select_node(key);
  return xpath != nullptr ? utl::parse<bool>(xpath.node().child_value())
                          : default_to;
}

pugi::xml_node get(pugi::xml_node const& node, char const* str) {
  auto const xpath = node.select_node(str);
  utl::verify(xpath != nullptr, "required xml node not found: {}", str);
  return xpath.node();
}

struct vdv_stop {
  explicit vdv_stop(location_idx_t const l, pugi::xml_node const n)
      : l_{l},
        dep_{get_opt_time(n, "Abfahrtszeit")},
        arr_{get_opt_time(n, "Ankunftszeit")},
        rt_dep_{get_opt_time(n, "IstAbfahrtPrognose")},
        rt_arr_{get_opt_time(n, "IstAnkunftPrognose")} {}

  std::pair<unixtime_t, event_type> get_event() const {
    if (dep_.has_value()) {
      return {*dep_, event_type::kDep};
    } else if (arr_.has_value()) {
      return {*arr_, event_type::kArr};
    } else {
      throw utl::fail("no event found (stop={})", l_);
    }
  }

  location_idx_t l_;
  std::optional<unixtime_t> dep_, arr_, rt_dep_, rt_arr_;
};

vector<vdv_stop> resolve_stops(timetable const& tt,
                               source_idx_t const src,
                               pugi::xml_node const run,
                               statistics& stats) {
  auto vdv_stops = vector<vdv_stop>{};

  for (auto const stop : run.select_nodes("IstHalt")) {
    ++stats.total_stops_;

    auto const vdv_stop_id =
        std::string_view{get(stop.node(), "HaltID").child_value()};
    auto const l = tt.locations_.find({vdv_stop_id, src});
    if (l.has_value()) {
      ++stats.resolved_stops_;

      if (get_opt_bool(stop.node(), "Zusatzhalt", false).value()) {
        ++stats.unsupported_additional_stops_;
        continue;
      }

      vdv_stops.emplace_back(l->l_, stop.node());
    } else {
      ++stats.unknown_stops_;
      std::cout << "could not resolve stop " << vdv_stop_id << ":\n";
      stop.node().print(std::cout);
      std::cout << "\n";
    }
  }

  return vdv_stops;
}

std::optional<rt::run> find_run(timetable const& tt,
                                pugi::xml_node const run,
                                auto const& vdv_stops,
                                statistics& stats) {
  using namespace std::literals;

  auto const vdv_line_id_xpath = run.select_node("LinienID");
  if (!vdv_line_id_xpath) {
    std::cout << "VDV run without line id:\n";
    return std::nullopt;
  }
  auto const vdv_line_id =
      std::string_view{vdv_line_id_xpath.node().child_value()};

  auto const vdv_direction_id_xpath = run.select_node("RichtungsID");
  if (!vdv_direction_id_xpath) {
    std::cout << "VDV run without direction id:\n";
    return std::nullopt;
  }

  auto const vdv_direction_id =
      std::stoul(vdv_direction_id_xpath.node().child_value());

  for (auto const& vdv_stop : vdv_stops) {
    for (auto const r : tt.location_routes_[vdv_stop.l_]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
        if (stop{s}.location_idx() != vdv_stop.l_) {
          continue;
        }

        auto const [t, ev_type] = vdv_stop.get_event();
        auto const [day_idx, mam] = tt.day_idx_mam(t);

        for (auto const& [ev_time_idx, ev_time] :
             utl::enumerate(tt.event_times_at_stop(
                 r, static_cast<stop_idx_t>(stop_idx), ev_type))) {
          if (ev_time.mam() != mam.count()) {
            continue;
          }

          auto const tr_day =
              static_cast<std::size_t>(to_idx(day_idx) - ev_time.days());
          auto const tr = tt.route_transport_ranges_[r][ev_time_idx];

          if (tt.bitfields_[tt.transport_traffic_days_[tr]].test(tr_day)) {
            auto const trip_line =
                size(tt.transport_section_lines_[tr]) == 1U
                    ? tt.trip_lines_[tt.transport_section_lines_[tr][0U]].view()
                    : tt.trip_lines_[tt.transport_section_lines_[tr][stop_idx]]
                          .view();

            if (vdv_line_id.find(trip_line.substr(0, trip_line.find(' '))) ==
                std::string_view::npos) {
              std::cout << "stop and event time match, but VDV line id "
                        << vdv_line_id
                        << " does not match GTFS route_short_name " << trip_line
                        << "\n";
              continue;
            }

            auto const trip_direction =
                tt.transport_to_trip_section_[tr].size() == 1U
                    ? tt.trip_direction_ids_
                          [tt.merged_trips_[tt.transport_to_trip_section_[tr]
                                                .front()]
                               .front()]
                    : tt.trip_direction_ids_
                          [tt.merged_trips_
                               [tt.transport_to_trip_section_[tr][stop_idx]]
                                   .front()];

            if ((vdv_direction_id == 1 && trip_direction == 1) ||
                (vdv_direction_id == 2 && trip_direction == 0)) {
              std::cout << "stop and event time match, but VDV direction id "
                        << vdv_direction_id
                        << " does not match GTFS direction id "
                        << trip_direction << "\n";
              continue;
            }

            std::cout << "matched run at vdv stop_idx = " << stop_idx
                      << ": VDV line id " << vdv_line_id
                      << " contains GTFS route_short_name " << trip_line
                      << "\n";

            return rt::run{transport{tr, day_idx_t{tr_day}},
                           {0U, static_cast<stop_idx_t>(location_seq.size())}};
          }
        }
      }
    }
    ++stats.no_transport_found_at_stop_;
  }
  return std::nullopt;
}

void update_run(timetable const& tt,
                rt_timetable& rtt,
                source_idx_t const src,
                run& r,
                auto const& vdv_stops,
                statistics& stats) {
  auto fr = rt::frun(tt, &rtt, r);
  if (!fr.is_rt()) {
    fr.rt_ = rtt.add_rt_transport(src, tt, r.t_);
  }

  auto delay = std::optional<duration_t>{};

  auto const update_event = [&](auto const stop_idx, auto const et,
                                auto const new_time) {
    delay = new_time - fr[stop_idx].scheduled_time(et);
    rtt.update_time(fr.rt_, stop_idx, et, new_time);
    rtt.dispatch_event_change(fr.t_, stop_idx, et, *delay, false);
    ++stats.updated_events_;
  };

  auto const propagate_delay = [&](auto const stop_idx, event_type et) {
    rtt.update_time(fr.rt_, stop_idx, et,
                    fr[stop_idx].scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, stop_idx, et, *delay, false);
    ++stats.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto const move_cursor = [&](auto const& vdv_stop) {
    if (cursor + 1 != vdv_stop) {
      for (auto it = cursor; it != vdv_stop - 1; ++it) {
        ++stats.skipped_vdv_stops_;
        std::cout << "skipped vdv stop: [id: " << tt.locations_.get(it->l_).id_
                  << ", name: " << tt.locations_.get(it->l_).name_ << "]\n";
      }
    }
    cursor = vdv_stop;
  };

  for (auto const stop_idx : fr.stop_range_) {
    auto matched = false;
    for (auto vdv_stop = cursor; vdv_stop != end(vdv_stops); ++vdv_stop) {
      if (fr[stop_idx].get_location_idx() == vdv_stop->l_) {
        matched = true;
        move_cursor(vdv_stop + 1);
        std::cout << "update at stop_idx = " << stop_idx
                  << ": [id: " << fr[stop_idx].id()
                  << ", name: " << fr[stop_idx].name() << "]\n";
        if (stop_idx != 0 && vdv_stop->rt_arr_.has_value()) {
          update_event(stop_idx, event_type::kArr, *vdv_stop->rt_arr_);
        }
        if (stop_idx != fr.stop_range_.to_ - 1 &&
            vdv_stop->rt_dep_.has_value()) {
          update_event(stop_idx, event_type::kDep, *vdv_stop->rt_dep_);
        }
        break;
      }
    }
    if (!matched && delay) {
      std::cout << "propagating delay at stop_idx = " << stop_idx
                << ": [id: " << fr[stop_idx].id()
                << ", name: " << fr[stop_idx].name() << "]\n";
      if (stop_idx != 0) {
        propagate_delay(stop_idx, event_type::kArr);
      }
      if (stop_idx != fr.stop_range_.to_ - 1) {
        propagate_delay(stop_idx, event_type::kDep);
      }
    }
  }

  while (cursor != end(vdv_stops)) {
    std::cout << "excess vdv stop: [id: " << tt.locations_.get(cursor->l_).id_
              << ", name: " << tt.locations_.get(cursor->l_).name_ << "]\n";
    ++stats.excess_vdv_stops_;
    ++cursor;
  }

  std::cout << "\n";
}

void process_vdv_run(timetable const& tt,
                     rt_timetable& rtt,
                     source_idx_t const src,
                     pugi::xml_node const run,
                     statistics& stats) {
  ++stats.total_runs_;

  auto vdv_stops = resolve_stops(tt, src, run, stats);

  auto r = find_run(tt, run, vdv_stops, stats);
  if (!r.has_value()) {
    ++stats.unmatchable_runs_;
    std::cout << "\nunmatchable run:\n";
    run.print(std::cout);
    std::cout << "\n";
    return;
  }
  ++stats.matched_runs_;

  update_run(tt, rtt, src, *r, vdv_stops, stats);
}

statistics vdv_update(timetable const& tt,
                      rt_timetable& rtt,
                      source_idx_t const src,
                      pugi::xml_document const& doc) {
  auto stats = statistics{};
  for (auto const& r :
       doc.select_nodes("DatenAbrufenAntwort/AUSNachricht/IstFahrt")) {
    if (get_opt_bool(r.node(), "Zusatzfahrt", false).value()) {
      ++stats.unsupported_additional_runs_;
      continue;
    } else if (get_opt_bool(r.node(), "FaelltAus", false).value()) {
      ++stats.unsupported_cancelled_runs_;
      continue;
    }

    process_vdv_run(tt, rtt, src, r.node(), stats);
  }
  return stats;
}

}  // namespace nigiri::rt::vdv
#include "nigiri/rt/vdv/vdv_update.h"

#include <sstream>
#include <string>
#include <string_view>

#include "pugixml.hpp"

#include "utl/enumerate.h"
#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/logging.h"
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
  return xpath ? std::optional{parse_time(xpath.node().child_value())}
               : std::nullopt;
}

std::optional<bool> get_opt_bool(
    pugi::xml_node const& node,
    char const* key,
    std::optional<bool> default_to = std::nullopt) {
  auto const xpath = node.select_node(key);
  return xpath ? utl::parse<bool>(xpath.node().child_value()) : default_to;
}

pugi::xml_node get(pugi::xml_node const& node, char const* str) {
  auto const xpath = node.select_node(str);
  utl::verify(xpath, "required xml node not found: {}", str);
  return xpath.node();
}

struct vdv_stop {
  explicit vdv_stop(pugi::xml_node const n)
      : id_{get(n, "HaltID").child_value()},
        dep_{get_opt_time(n, "Abfahrtszeit")},
        arr_{get_opt_time(n, "Ankunftszeit")},
        rt_dep_{get_opt_time(n, "IstAbfahrtPrognose")},
        rt_arr_{get_opt_time(n, "IstAnkunftPrognose")},
        is_additional_{get_opt_bool(n, "Zusatzhalt", false).value()} {}

  std::pair<unixtime_t, event_type> get_event() const {
    if (dep_.has_value()) {
      return {*dep_, event_type::kDep};
    } else if (arr_.has_value()) {
      return {*arr_, event_type::kArr};
    } else {
      throw utl::fail("no event found (stop={})", id_);
    }
  }

  std::string_view id_;
  std::optional<unixtime_t> dep_, arr_, rt_dep_, rt_arr_;
  bool is_additional_;
};

std::optional<rt::run> get_run(timetable const& tt,
                               source_idx_t const src,
                               auto const& vdv_stops) {

  auto const first_it =
      utl::find_if(vdv_stops, [](auto&& s) { return !s.is_additional_; });
  if (first_it == end(vdv_stops)) {
    return std::nullopt;
  }

  auto const& first_stop = *first_it;
  auto const l = tt.locations_.get({first_stop.id_, src}).l_;

  for (auto const r : tt.location_routes_[l]) {
    auto const location_seq = tt.route_location_seq_[r];
    for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
      if (stop{s}.location_idx() != l) {
        continue;
      }

      auto const [t, ev_type] = first_stop.get_event();
      auto const [day_idx, mam] = tt.day_idx_mam(t);
      auto const event_times = tt.event_times_at_stop(
          r, static_cast<stop_idx_t>(stop_idx), event_type::kDep);
      auto const it = utl::find_if(event_times, [&](delta const ev_time) {
        return ev_time.mam() == mam.count();
      });
      if (it == end(event_times)) {
        continue;
      }

      auto const ev_day_offset = it->days();
      auto const start_day =
          static_cast<std::size_t>(to_idx(day_idx) - ev_day_offset);
      auto const tr = tt.route_transport_ranges_[r][static_cast<size_t>(
          std::distance(begin(event_times), it))];
      if (tt.bitfields_[tt.transport_traffic_days_[tr]].test(start_day)) {
        return rt::run{transport{tr, day_idx_t{start_day}},
                       {0U, static_cast<stop_idx_t>(location_seq.size())}};
      }
    }
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
  };

  auto const propagate_delay = [&](auto const stop_idx, event_type et) {
    rtt.update_time(fr.rt_, stop_idx, et,
                    fr[stop_idx].scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, stop_idx, et, *delay, false);
  };

  auto vdv_stop_it = begin(vdv_stops);

  for (auto const stop_idx : fr.stop_range_) {

    while (vdv_stop_it != end(vdv_stops) && vdv_stop_it->is_additional_) {
      ++stats.unsupported_additional_stop_;
      ++vdv_stop_it;
    }

    // match stop ids
    if (vdv_stop_it != end(vdv_stops) &&
        vdv_stop_it->id_ == fr[stop_idx].id()) {
      if (stop_idx != 0 && vdv_stop_it->rt_arr_.has_value()) {
        update_event(stop_idx, event_type::kArr, *vdv_stop_it->rt_arr_);
      }
      if (stop_idx != fr.stop_range_.to_ - 1 &&
          vdv_stop_it->rt_dep_.has_value()) {
        update_event(stop_idx, event_type::kDep, *vdv_stop_it->rt_dep_);
      }
      ++vdv_stop_it;
      // propagate delay
    } else if (delay) {
      if (stop_idx != 0) {
        propagate_delay(stop_idx, event_type::kArr);
      }
      if (stop_idx != fr.stop_range_.to_ - 1) {
        propagate_delay(stop_idx, event_type::kDep);
      }
    }
  }
}

void process_vdv_run(timetable const& tt,
                     rt_timetable& rtt,
                     source_idx_t const src,
                     pugi::xml_node const run_node,
                     statistics& stats) {
  auto const vdv_stops = utl::to_vec(
      run_node.select_nodes("IstHalt"),
      [](auto&& stop_xpath) { return vdv_stop{stop_xpath.node()}; });

  auto r = get_run(tt, src, vdv_stops);
  if (!r.has_value()) {
    ++stats.unmatchable_run_;
    return;
  }

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
      ++stats.unsupported_additional_run;
      continue;
    } else if (get_opt_bool(r.node(), "FaelltAus", false).value()) {
      ++stats.unsupported_cancelled_run;
      continue;
    }

    process_vdv_run(tt, rtt, src, r.node(), stats);
  }
  return stats;
}

}  // namespace nigiri::rt::vdv
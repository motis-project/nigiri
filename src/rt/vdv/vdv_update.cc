#include "nigiri/rt/vdv/vdv_update.h"

#include <ranges>
#include <sstream>
#include <string>
#include <string_view>

#include "pugixml.hpp"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "nigiri/routing/for_each_meta.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt::vdv {

#define VDV_DEBUG
#ifdef VDV_DEBUG
#define vdv_trace(s) std::cout << s << "\n"
#else
#define vdv_trace(s)
#endif

unixtime_t parse_time(std::string const& str) {
  unixtime_t parsed;
  auto ss = std::stringstream{str};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

std::optional<unixtime_t> updater::get_opt_time(pugi::xml_node const& node,
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

vector<updater::vdv_stop> updater::resolve_stops(pugi::xml_node const vdv_run) {
  auto vdv_stops = vector<vdv_stop>{};

  for (auto const stop : vdv_run.select_nodes("IstHalt")) {
    ++stats_.total_stops_;

    auto const vdv_stop_id =
        std::string_view{get(stop.node(), "HaltID").child_value()};
    auto const l = tt_.locations_.find({vdv_stop_id, src_idx_});

    if (get_opt_bool(stop.node(), "Zusatzhalt", false).value()) {
      ++stats_.unsupported_additional_stops_;
      vdv_trace(std::format("unsupported additional stop: [id: {}, name: {}]",
                            vdv_stop_id,
                            l.has_value() ? l->name_ : "unresolvable"));
    }

    if (l.has_value()) {
      ++stats_.resolved_stops_;
      vdv_stops.emplace_back(l->l_, vdv_stop_id, stop.node());
    } else {
      ++stats_.unknown_stops_;
      vdv_stops.emplace_back(location_idx_t::invalid(), vdv_stop_id,
                             stop.node());
      vdv_trace(std::format("unresolvable stop: {}", vdv_stop_id));
    }
  }

  return vdv_stops;
}

struct candidate {
  explicit candidate(run const& r, std::uint32_t const total_length)
      : r_{r}, total_length_{total_length} {}

  void finish_stop() {
    score_ += local_score_;
    local_score_ = 0.0;
  }

  friend bool operator<(candidate const& a, candidate const& b) {
    return a.score_ > b.score_ ||
           (a.score_ == b.score_ && a.total_length_ < b.total_length_);
  }

  friend bool operator==(candidate const& a, candidate const& b) {
    return a.score_ == b.score_ && a.total_length_ == b.total_length_;
  }

  run r_;
  double score_{0.0};
  double local_score_{0.0};
  std::uint32_t total_length_;
};

std::optional<rt::run> updater::find_run(std::string_view vdv_run_id,
                                         vector<vdv_stop> const& vdv_stops,
                                         bool const is_complete_run) {
  if (!is_complete_run) {
    ++stats_.search_on_incomplete_;
    vdv_trace(std::format("Attempting to match an incomplete vdv run: {}",
                          vdv_run_id));
  }

  auto candidates = std::vector<candidate>{};

  for (auto const& vdv_stop : vdv_stops) {
    if (vdv_stop.l_ == location_idx_t::invalid()) {
      continue;
    }
    auto no_transport_found_at_stop = true;
    for (auto const l : tt_.locations_.equivalences_[vdv_stop.l_]) {
      for (auto const r : tt_.location_routes_[l]) {
        auto const location_seq = tt_.route_location_seq_[r];
        for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
          if (!matches(tt_, routing::location_match_mode::kEquivalent,
                       stop{s}.location_idx(), vdv_stop.l_)) {
            continue;
          }

          auto ev = vdv_stop.get_event(event_type::kDep);
          if (stop_idx == location_seq.size() - 1) {
            ev = vdv_stop.get_event(event_type::kArr);
          }
          if (!ev.has_value()) {
            continue;
          }

          auto const [t, ev_type] = *ev;
          auto const [day_idx, mam] = tt_.day_idx_mam(t);

          for (auto const [ev_time_idx, ev_time] :
               utl::enumerate(tt_.event_times_at_stop(
                   r, static_cast<stop_idx_t>(stop_idx), ev_type))) {

            auto const tr =
                transport{tt_.route_transport_ranges_[r][ev_time_idx],
                          day_idx - day_idx_t{ev_time.days()}};

            if (tt_.bitfields_[tt_.transport_traffic_days_[tr.t_idx_]].test(
                    to_idx(tr.day_))) {

              auto candidate =
                  std::find_if(begin(candidates), end(candidates),
                               [&](auto const& c) { return c.r_.t_ == tr; });

              if (candidate != end(candidates) &&
                  stop_idx < candidate->r_.stop_range_.from_) {
                continue;
              }

              if (candidate == end(candidates)) {
                candidates.emplace_back(
                    run{tr,
                        interval{static_cast<stop_idx_t>(stop_idx),
                                 static_cast<stop_idx_t>(location_seq.size())}},
                    location_seq.size());
                candidate = end(candidates) - 1;
              }

              candidate->local_score_ = std::max(
                  candidate->local_score_,
                  1.0 / static_cast<double>(
                            std::abs(ev_time.mam() - mam.count()) + 1));

              no_transport_found_at_stop = false;
            }
          }
        }
      }
    }
    if (no_transport_found_at_stop) {
      ++stats_.no_transport_found_at_stop_;
    }
    for (auto& c : candidates) {
      c.finish_stop();
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(begin(candidates), end(candidates));

  if (candidates.front().score_ < vdv_stops.size() / 2.0) {
    return std::nullopt;
  }

  if (candidates.size() > 1 && candidates[0] == candidates[1]) {
#ifdef VDV_DEBUG
    vdv_trace(std::format("multiple match candidates for {}:", vdv_run_id));
    for (auto const& c : candidates) {
      vdv_trace(std::format(
          "[line: {}, score: {}, length: {}]",
          tt_.trip_lines_
              [tt_.transport_section_lines_[c.r_.t_.t_idx_].size() == 1
                   ? tt_.transport_section_lines_[c.r_.t_.t_idx_].front()
                   : tt_.transport_section_lines_[c.r_.t_.t_idx_]
                                                 [c.r_.stop_range_.from_]]
                  .view(),
          c.score_, c.total_length_));
    }
#endif
    ++stats_.multiple_matches_;
    return std::nullopt;
  }

  ++stats_.found_runs_;
  vdv_nigiri_[vdv_run_id] = candidates.front().r_;
  return candidates.front().r_;
}

void monotonize(frun& fr, rt_timetable& rtt) {
  auto next_time = unixtime_t::max();

  auto const update = [&](auto const& rs, auto const ev, auto const new_time) {
    rtt.update_time(fr.rt_, rs.stop_idx_, ev, new_time);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, ev,
                              new_time - rs.scheduled_time(ev), false);
    next_time = new_time;
  };

  for (auto it = rbegin(fr); it != rend(fr); --it) {
    if (it.rs_.stop_idx_ != fr.stop_range_.to_ - 1) {
      update(it.rs_, event_type::kDep,
             std::min(it.rs_.time(event_type::kDep), next_time));
    }
    if (it.rs_.stop_idx_ != fr.stop_range_.from_) {
      update(it.rs_, event_type::kArr,
             std::min(it.rs_.time(event_type::kArr), next_time));
    }
  }
}

void updater::update_run(rt_timetable& rtt,
                         run& r,
                         vector<vdv_stop> const& vdv_stops,
                         bool const is_complete_run) {
  auto fr = rt::frun(tt_, &rtt, r);
  if (!fr.is_rt()) {
    fr.rt_ = rtt.add_rt_transport(src_idx_, tt_, r.t_);
  }

  auto delay = std::optional<duration_t>{};

  auto const update_event = [&](auto const& rs, auto const et,
                                auto const new_time) {
    delay = new_time - rs.scheduled_time(et);
    vdv_trace(std::format(
        "update [stop_idx: {}, id: {}, name: {}] {}: {}{}{}", rs.stop_idx_,
        rs.id(), rs.name(), et == event_type::kArr ? "ARR" : "DEP",
        rs.scheduled_time(et), delay->count() >= 0 ? "+" : "", delay->count()));
    rtt.update_time(fr.rt_, rs.stop_idx_, et, new_time);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats_.updated_events_;
  };

  auto const propagate_delay = [&](auto const& rs, event_type et) {
    vdv_trace(std::format(
        "propagate [stop_idx: {}, id: {}, name: {}] {}: {}{}{}", rs.stop_idx_,
        rs.id(), rs.name(), et == event_type::kArr ? "ARR" : "DEP",
        rs.scheduled_time(et), delay->count() >= 0 ? "+" : "", delay->count()));
    rtt.update_time(fr.rt_, rs.stop_idx_, et, rs.scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats_.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto skipped_stops = std::vector<vdv_stop>{};
  auto const print_skipped_stops = [&]() {
    for (auto const& s [[maybe_unused]] : skipped_stops) {
      ++stats_.skipped_vdv_stops_;
      vdv_trace(std::format("skipped vdv stop: [id: {}, name: {}]", s.id_,
                            s.l_ == location_idx_t::invalid()
                                ? "unresolvable"
                                : tt_.locations_.get(s.l_).name_));
    }
  };

  vdv_trace(std::format("---updating {}, stop_range: [{}, {}[", fr.name(),
                        fr.stop_range_.from_, fr.stop_range_.to_));
  for (auto const rs : fr) {
    auto matched_arr = false;
    auto matched_dep = false;
    skipped_stops.clear();
    for (auto vdv_stop = cursor; vdv_stop != end(vdv_stops); ++vdv_stop) {
      if (vdv_stop->l_ != location_idx_t::invalid() &&
          matches(tt_, routing::location_match_mode::kEquivalent,
                  rs.get_location_idx(), vdv_stop->l_)) {
        if (rs.stop_idx_ != 0 && vdv_stop->arr_.has_value() &&
            (static_cast<std::uint32_t>(
                 std::abs(vdv_stop->arr_.value().time_since_epoch().count() -
                          rs.scheduled_time(event_type::kArr)
                              .time_since_epoch()
                              .count())) <= kAllowedTimeDiscrepancy)) {
          matched_arr = true;
          if (vdv_stop->rt_arr_.has_value()) {
            update_event(rs, event_type::kArr, *vdv_stop->rt_arr_);
          }
        }
        if (rs.stop_idx_ != fr.stop_range_.to_ - 1 &&
            vdv_stop->dep_.has_value() &&
            static_cast<std::uint32_t>(
                std::abs(vdv_stop->dep_.value().time_since_epoch().count() -
                         rs.scheduled_time(event_type::kDep)
                             .time_since_epoch()
                             .count())) <= kAllowedTimeDiscrepancy) {
          matched_dep = true;
          if (vdv_stop->rt_dep_.has_value()) {
            update_event(rs, event_type::kDep, *vdv_stop->rt_dep_);
          }
        }
        if (matched_arr || matched_dep) {
          cursor = vdv_stop + 1;
          print_skipped_stops();
          break;
        }
      }
      skipped_stops.emplace_back(*vdv_stop);
    }
    if (!matched_arr && !matched_dep && is_complete_run &&
        rs.stop_idx_ < vdv_stops.size()) {
      vdv_trace(
          std::format("missing gtfs stop at stop_idx = {}: [id: {}, name: {}]",
                      rs.stop_idx_, rs.id(), rs.name()));
    }
    if (delay) {
      if (rs.stop_idx_ != 0 && !matched_arr) {
        propagate_delay(rs, event_type::kArr);
      }
      if (rs.stop_idx_ != fr.stop_range_.to_ - 1 && !matched_dep) {
        propagate_delay(rs, event_type::kDep);
      }
    }
  }

  while (cursor != end(vdv_stops)) {
    vdv_trace(std::format("excess vdv stop: [id: {}, name: {}]", cursor->id_,
                          cursor->l_ == location_idx_t::invalid()
                              ? "unresolvable"
                              : tt_.locations_.get(cursor->l_).name_));
    ++stats_.excess_vdv_stops_;
    ++cursor;
  }

  monotonize(fr, rtt);
}

void updater::process_vdv_run(rt_timetable& rtt, pugi::xml_node const vdv_run) {
  ++stats_.total_runs_;

  auto vdv_stops = resolve_stops(vdv_run);

  auto const vdv_run_id =
      std::string{
          get(vdv_run, "./FahrtRef/FahrtID/FahrtBezeichner").child_value()}
          .append(get(vdv_run, "./FahrtRef/FahrtID/Betriebstag").child_value());

  if (vdv_stops.empty()) {
    ++stats_.runs_without_stops_;
    vdv_trace(std::format("vdv run without stops: {}", vdv_run_id));
    return;
  }

  auto const is_complete_run = *get_opt_bool(vdv_run, "Komplettfahrt", false);

  auto r = vdv_nigiri_.contains(vdv_run_id)
               ? std::optional{vdv_nigiri_.at(vdv_run_id)}
               : find_run(vdv_run_id, vdv_stops, is_complete_run);
  if (!r.has_value()) {
#ifdef VDV_DEBUG
    vdv_trace("unmatchable run:");
    vdv_run.print(std::cout);
    vdv_trace("\n");
#endif
    ++stats_.unmatchable_runs_;
    return;
  }
  ++stats_.matched_runs_;

  update_run(rtt, *r, vdv_stops, is_complete_run);
}

void updater::update(rt_timetable& rtt, pugi::xml_document const& doc) {
  for (auto const& vdv_run : doc.select_nodes("//IstFahrt")) {
    if (get_opt_bool(vdv_run.node(), "Zusatzfahrt", false).value()) {
#ifdef VDV_DEBUG
      vdv_trace("unsupported additional run:");
      vdv_run.node().print(std::cout);
#endif
      ++stats_.unsupported_additional_runs_;
      continue;
    } else if (get_opt_bool(vdv_run.node(), "FaelltAus", false).value()) {
#ifdef VDV_DEBUG
      vdv_trace("unsupported canceled run:");
      vdv_run.node().print(std::cout);
#endif
      ++stats_.unsupported_cancelled_runs_;
      continue;
    }

    process_vdv_run(rtt, vdv_run.node());
  }
}

}  // namespace nigiri::rt::vdv
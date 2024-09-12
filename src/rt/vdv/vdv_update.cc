#include "nigiri/rt/vdv/vdv_update.h"

#include <sstream>
#include <string>
#include <string_view>

#include "pugixml.hpp"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "fmt/core.h"

#include "nigiri/common/mam_dist.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt::vdv {

// #define VDV_DEBUG
#ifdef VDV_DEBUG
#define vdv_trace(...) fmt::print(__VA_ARGS__)
#else
#define vdv_trace(...)
#endif

std::ostream& operator<<(std::ostream& out, statistics const& s) {
  out << "unsupported additional runs: " << s.unsupported_additional_runs_
      << "\nunsupported cancelled runs: " << s.unsupported_cancelled_runs_
      << "\ntotal stops: " << s.total_stops_
      << "\nresolved stops: " << s.resolved_stops_
      << "\nunknown stops: " << s.unknown_stops_
      << "\nunsupported additional stops: " << s.unsupported_additional_stops_
      << "\nno transport found at stop: " << s.no_transport_found_at_stop_
      << "\nsearches on incomplete runs: " << s.search_on_incomplete_
      << "\nfound runs: " << s.found_runs_
      << "\nmultiple matches: " << s.multiple_matches_
      << "\ntotal runs: " << s.total_runs_
      << "\nmatched runs: " << s.matched_runs_
      << "\nunmatchable runs: " << s.unmatchable_runs_
      << "\nruns without stops: " << s.runs_without_stops_
      << "\nskipped vdv stops: " << s.skipped_vdv_stops_
      << "\nexcess vdv stops: " << s.excess_vdv_stops_
      << "\nupdated events: " << s.updated_events_
      << "\npropagated delays: " << s.propagated_delays_ << "\n";
  return out;
}

statistics& operator+=(statistics& lhs, statistics const& rhs) {
  lhs.unsupported_additional_runs_ += rhs.unsupported_additional_runs_;
  lhs.unsupported_cancelled_runs_ += rhs.unsupported_cancelled_runs_;
  lhs.total_stops_ += rhs.total_stops_;
  lhs.resolved_stops_ += rhs.resolved_stops_;
  lhs.unknown_stops_ += rhs.unknown_stops_;
  lhs.unsupported_additional_stops_ += rhs.unsupported_additional_stops_;
  lhs.total_runs_ += rhs.total_runs_;
  lhs.no_transport_found_at_stop_ += rhs.no_transport_found_at_stop_;
  lhs.search_on_incomplete_ += rhs.search_on_incomplete_;
  lhs.found_runs_ += rhs.found_runs_;
  lhs.multiple_matches_ += rhs.multiple_matches_;
  lhs.matched_runs_ += rhs.matched_runs_;
  lhs.unmatchable_runs_ += rhs.unmatchable_runs_;
  lhs.runs_without_stops_ += rhs.runs_without_stops_;
  lhs.skipped_vdv_stops_ += rhs.skipped_vdv_stops_;
  lhs.excess_vdv_stops_ += rhs.excess_vdv_stops_;
  lhs.updated_events_ += rhs.updated_events_;
  lhs.propagated_delays_ += rhs.propagated_delays_;
  return lhs;
}

updater::updater(nigiri::timetable const& tt, source_idx_t const src_idx)
    : tt_{tt}, src_idx_{src_idx} {}

void updater::reset_vdv_run_ids_() { vdv_nigiri_.clear(); }

statistics const& updater::get_stats() const { return stats_; }

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

updater::vdv_stop::vdv_stop(nigiri::location_idx_t const l,
                            std::string_view id,
                            pugi::xml_node const n)
    : l_{l},
      id_{id},
      dep_{get_opt_time(n, "Abfahrtszeit")},
      arr_{get_opt_time(n, "Ankunftszeit")},
      rt_dep_{get_opt_time(n, "IstAbfahrtPrognose")},
      rt_arr_{get_opt_time(n, "IstAnkunftPrognose")} {}

std::optional<std::pair<unixtime_t, event_type>> updater::vdv_stop::get_event(
    event_type et) const {
  if (et == event_type::kArr && arr_.has_value()) {
    return std::pair{*arr_, event_type::kArr};
  } else if (et == event_type::kDep && dep_.has_value()) {
    return std::pair{*dep_, event_type::kDep};
  } else {
    return std::nullopt;
  }
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
      vdv_trace("unsupported additional stop: [id: {}, name: {}]\n",
                vdv_stop_id, l.has_value() ? l->name_ : "unresolvable");
    }

    if (l.has_value()) {
      ++stats_.resolved_stops_;
      vdv_stops.emplace_back(l->l_, vdv_stop_id, stop.node());
    } else {
      ++stats_.unknown_stops_;
      vdv_stops.emplace_back(location_idx_t::invalid(), vdv_stop_id,
                             stop.node());
      vdv_trace("unresolvable stop: {}\n", vdv_stop_id);
    }
  }

  return vdv_stops;
}

struct candidate {
  explicit candidate(run const& r, std::uint32_t const total_length)
      : r_{r}, total_length_{total_length} {}

  void finish_stop() {
    score_ += local_best_;
    local_best_ = 0U;
  }

  friend bool operator<(candidate const& a, candidate const& b) {
    return a.score_ > b.score_ ||
           (a.score_ == b.score_ && a.total_length_ < b.total_length_);
  }

  friend bool operator==(candidate const& a, candidate const& b) {
    return a.score_ == b.score_ && a.total_length_ == b.total_length_;
  }

  run r_;
  std::uint32_t score_{0U};
  std::uint32_t local_best_{0U};
  std::uint32_t total_length_;
};

std::optional<rt::run> updater::find_run(std::string_view vdv_run_id,
                                         vector<vdv_stop> const& vdv_stops,
                                         bool const is_complete_run) {
  if (!is_complete_run) {
    ++stats_.search_on_incomplete_;
    vdv_trace("Attempting to match an incomplete vdv run: {}\n", vdv_run_id);
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

          auto const vdv_ev = stop_idx == location_seq.size() - 1
                                  ? vdv_stop.get_event(event_type::kArr)
                                  : vdv_stop.get_event(event_type::kDep);
          if (!vdv_ev.has_value()) {
            continue;
          }

          auto const [vdv_time, ev_type] = *vdv_ev;
          auto const [vdv_day_idx, vdv_mam] = tt_.day_idx_mam(vdv_time);

          for (auto const [nigiri_ev_time_idx, nigiri_ev_time] :
               utl::enumerate(tt_.event_times_at_stop(
                   r, static_cast<stop_idx_t>(stop_idx), ev_type))) {
            auto const [error, day_shift] =
                mam_dist(vdv_mam, i32_minutes{nigiri_ev_time.mam()});
            auto const local_score =
                kExactMatchScore - error.count() * error.count();
            if (local_score < 0) {
              continue;
            }

            auto const tr = transport{
                tt_.route_transport_ranges_[r][nigiri_ev_time_idx],
                vdv_day_idx -
                    day_idx_t{nigiri_ev_time.days() + day_shift.count()}};

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

              candidate->local_best_ =
                  std::max(candidate->local_best_,
                           static_cast<std::uint32_t>(local_score));

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

  if (candidates.front().score_ < vdv_stops.size() * kExactMatchScore / 2) {
    return std::nullopt;
  }

  if (candidates.size() > 1 && candidates[0] == candidates[1]) {
#ifdef VDV_DEBUG
    vdv_trace("multiple match candidates for {}:\n", vdv_run_id);
    for (auto const& c : candidates) {
      vdv_trace("[line: {}, score: {}, length: {}]\n",
                tt_.trip_lines_
                    [tt_.transport_section_lines_[c.r_.t_.t_idx_].size() == 1
                         ? tt_.transport_section_lines_[c.r_.t_.t_idx_].front()
                         : tt_.transport_section_lines_[c.r_.t_.t_idx_]
                                                       [c.r_.stop_range_.from_]]
                        .view(),
                c.score_, c.total_length_);
    }
#endif
    ++stats_.multiple_matches_;
    return std::nullopt;
  }
#ifdef VDV_DEBUG
  else {
    vdv_trace("match candidates for {}:\n", vdv_run_id);
    for (auto const& c : candidates) {
      vdv_trace("[line: {}, score: {}, length: {}]\n",
                tt_.trip_lines_
                    [tt_.transport_section_lines_[c.r_.t_.t_idx_].size() == 1
                         ? tt_.transport_section_lines_[c.r_.t_.t_idx_].front()
                         : tt_.transport_section_lines_[c.r_.t_.t_idx_]
                                                       [c.r_.stop_range_.from_]]
                        .view(),
                c.score_, c.total_length_);
    }
  }
#endif

  ++stats_.found_runs_;
  vdv_nigiri_[vdv_run_id] = candidates.front().r_;
  return candidates.front().r_;
}

void update_event(rt_timetable& rtt,
                  frun::run_stop const& rs,
                  event_type const et,
                  unixtime_t const new_time,
                  std::optional<duration_t>* delay_propagation = nullptr) {
  auto delay = new_time - rs.scheduled_time(et);
  vdv_trace("update [stop_idx: {}, id: {}, name: {}] {}: {}{}{}\n",
            rs.stop_idx_, rs.id(), rs.name(),
            et == event_type::kArr ? "ARR" : "DEP", rs.scheduled_time(et),
            delay.count() >= 0 ? "+" : "", delay.count());
  rtt.update_time(rs.fr_->rt_, rs.stop_idx_, et, new_time);
  rtt.dispatch_event_change(rs.fr_->t_, rs.stop_idx_, et, delay, false);
  if (delay_propagation != nullptr) {
    *delay_propagation = delay;
  }
}

void monotonize(frun& fr, rt_timetable& rtt) {
  vdv_trace("---monotonizing {}, stop_range: [{}, {}[\n", fr.name(),
            fr.stop_range_.from_, fr.stop_range_.to_);

  auto upper_bound = unixtime_t::max();
  for (auto const rs : it_range(rbegin(fr), rend(fr))) {
    if (rs.stop_idx_ != fr.stop_range_.to_ - 1) {
      upper_bound = std::min(rs.time(event_type::kDep), upper_bound);
      update_event(rtt, rs, event_type::kDep, upper_bound);
    }
    if (rs.stop_idx_ != fr.stop_range_.from_) {
      upper_bound = std::min(rs.time(event_type::kArr), upper_bound);
      update_event(rtt, rs, event_type::kArr, upper_bound);
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

  auto const propagate_delay = [&](auto const& rs, event_type et) {
    vdv_trace("propagate [stop_idx: {}, id: {}, name: {}] {}: {}{}{}\n",
              rs.stop_idx_, rs.id(), rs.name(),
              et == event_type::kArr ? "ARR" : "DEP", rs.scheduled_time(et),
              delay->count() >= 0 ? "+" : "", delay->count());
    rtt.update_time(fr.rt_, rs.stop_idx_, et, rs.scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats_.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto skipped_stops = std::vector<vdv_stop>{};
  auto const print_skipped_stops = [&]() {
    for (auto const& s [[maybe_unused]] : skipped_stops) {
      ++stats_.skipped_vdv_stops_;
      vdv_trace("skipped vdv stop: [id: {}, name: {}]\n", s.id_,
                s.l_ == location_idx_t::invalid()
                    ? "unresolvable"
                    : tt_.locations_.get(s.l_).name_);
    }
  };

  vdv_trace("---updating {}, stop_range: [{}, {}[\n", fr.name(),
            fr.stop_range_.from_, fr.stop_range_.to_);
  for (auto const rs : fr) {
    auto matched_arr = false;
    auto matched_dep = false;
    skipped_stops.clear();
    for (auto vdv_stop = cursor; vdv_stop != end(vdv_stops); ++vdv_stop) {
      if (vdv_stop->l_ != location_idx_t::invalid() &&
          matches(tt_, routing::location_match_mode::kEquivalent,
                  rs.get_location_idx(), vdv_stop->l_)) {
        if (rs.stop_idx_ != 0 && vdv_stop->arr_.has_value() &&
            (static_cast<std::uint32_t>(std::abs(
                 (vdv_stop->arr_.value() - rs.scheduled_time(event_type::kArr))
                     .count())) <= kAllowedTimeDiscrepancy)) {
          matched_arr = true;
          if (vdv_stop->rt_arr_.has_value()) {
            update_event(rtt, rs, event_type::kArr, *vdv_stop->rt_arr_, &delay);
            ++stats_.updated_events_;
          }
        }
        if (rs.stop_idx_ != fr.stop_range_.to_ - 1 &&
            vdv_stop->dep_.has_value() &&
            static_cast<std::uint32_t>(std::abs(
                (vdv_stop->dep_.value() - rs.scheduled_time(event_type::kDep))
                    .count())) <= kAllowedTimeDiscrepancy) {
          matched_dep = true;
          if (vdv_stop->rt_dep_.has_value()) {
            update_event(rtt, rs, event_type::kDep, *vdv_stop->rt_dep_, &delay);
            ++stats_.updated_events_;
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
      vdv_trace("missing gtfs stop at stop_idx = {}: [id: {}, name: {}]\n",
                rs.stop_idx_, rs.id(), rs.name());
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
    vdv_trace("excess vdv stop: [id: {}, name: {}]\n", cursor->id_,
              cursor->l_ == location_idx_t::invalid()
                  ? "unresolvable"
                  : tt_.locations_.get(cursor->l_).name_);
    ++stats_.excess_vdv_stops_;
    ++cursor;
  }

  monotonize(fr, rtt);
}

void updater::process_vdv_run(rt_timetable& rtt, pugi::xml_node const vdv_run) {
  ++stats_.total_runs_;

  auto vdv_stops = resolve_stops(vdv_run);

  auto const vdv_run_id = fmt::format(
      "{}{}", get(vdv_run, "./FahrtRef/FahrtID/FahrtBezeichner").child_value(),
      get(vdv_run, "./FahrtRef/FahrtID/Betriebstag").child_value());

  if (vdv_stops.empty()) {
    ++stats_.runs_without_stops_;
    vdv_trace("vdv run without stops: {}\n", vdv_run_id);
    return;
  }

  auto const is_complete_run = *get_opt_bool(vdv_run, "Komplettfahrt", false);

  auto r = vdv_nigiri_.contains(vdv_run_id)
               ? std::optional{vdv_nigiri_.at(vdv_run_id)}
               : find_run(vdv_run_id, vdv_stops, is_complete_run);
  if (!r.has_value()) {
    if (is_complete_run) {
#ifdef VDV_DEBUG
      vdv_trace("unmatchable run:\n");
      vdv_run.print(std::cout);
      vdv_trace("\n");
#endif
      ++stats_.unmatchable_runs_;
    }
    return;
  }
  ++stats_.matched_runs_;

  update_run(rtt, *r, vdv_stops, is_complete_run);
}

void updater::update(rt_timetable& rtt, pugi::xml_document const& doc) {
  for (auto const& vdv_run : doc.select_nodes("//IstFahrt")) {
    if (get_opt_bool(vdv_run.node(), "Zusatzfahrt", false).value()) {
#ifdef VDV_DEBUG
      vdv_trace("unsupported additional run:\n");
      vdv_run.node().print(std::cout);
#endif
      ++stats_.unsupported_additional_runs_;
      continue;
    } else if (get_opt_bool(vdv_run.node(), "FaelltAus", false).value()) {
#ifdef VDV_DEBUG
      vdv_trace("unsupported canceled run:\n");
      vdv_run.node().print(std::cout);
#endif
      ++stats_.unsupported_cancelled_runs_;
      continue;
    }

    process_vdv_run(rtt, vdv_run.node());
  }
}

}  // namespace nigiri::rt::vdv
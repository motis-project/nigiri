#include "nigiri/rt/vdv_aus.h"

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

using namespace std::chrono_literals;

namespace nigiri::rt::vdv_aus {

constexpr auto const kExactMatchScore = 230;
constexpr auto const kFirstMatchThreshold = 0.5;
constexpr auto const kFirstMatchThresholdIncomplete = 0.7;
constexpr auto const kAdditionalMatchThreshold = 0.975;
constexpr auto const kAllowedTimeDiscrepancy = []() {
  auto error = 0;
  while (kExactMatchScore - error * error > 0) {
    ++error;
  }
  return error - 1;
}();  // minutes
constexpr auto const kCleanUpInterval = 12h;
constexpr auto const kMatchRetention = 48h;

// #define VDV_DEBUG
#ifdef VDV_DEBUG
#define vdv_trace(...) fmt::print(__VA_ARGS__)
#else
#define vdv_trace(...)
#endif

std::ostream& operator<<(std::ostream& out, statistics const& s) {
  out << "found runs: " << s.found_runs_ << " / " << s.total_runs_ << " ("
      << (100.0 * static_cast<double>(s.found_runs_) / s.total_runs_) << "%)\n"
      << "unsupported additional runs: " << s.unsupported_additional_runs_
      << "\nunsupported additional stops: " << s.unsupported_additional_stops_
      << "\ncurrent matches total: " << s.current_matches_total_
      << "\ncurrent matches non-empty: " << s.current_matches_non_empty_
      << "\ntotal runs: " << s.total_runs_
      << "\ncomplete runs: " << s.complete_runs_
      << "\nunique runs: " << s.unique_runs_
      << "\nmatch attempts: " << s.match_attempts_
      << "\nmatched runs: " << s.matched_runs_
      << "\nmultiple matches: " << s.multiple_matches_
      << "\nincomplete not seen before: " << s.incomplete_not_seen_before_
      << "\ncomplete after incomplete: " << s.complete_after_incomplete_
      << "\nno transport found at stop: " << s.no_transport_found_at_stop_
      << "\ntotal stops: " << s.total_stops_
      << "\nresolved stops: " << s.resolved_stops_
      << "\nruns without stops: " << s.runs_without_stops_
      << "\ncancelled runs: " << s.cancelled_runs_
      << "\nskipped vdv stops: " << s.skipped_vdv_stops_
      << "\nexcess vdv stops: " << s.excess_vdv_stops_
      << "\nupdated events: " << s.updated_events_
      << "\npropagated delays: " << s.propagated_delays_ << "\n";
  return out;
}

statistics& statistics::operator+=(statistics const& o) {
  auto const x = cista::to_tuple(*this);
  auto const y = cista::to_tuple(o);
  auto const add = [](auto& a, auto const b) {
    if constexpr (std::is_same_v<std::uint32_t, std::decay_t<decltype(b)>>) {
      a += b;
    }
  };
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((add(std::get<I>(x), std::get<I>(y))), ...);
  }(std::make_index_sequence<std::tuple_size_v<decltype(x)>>());
  return *this;
}

updater::updater(nigiri::timetable const& tt,
                 source_idx_t const src_idx,
                 xml_format const format)
    : tt_{tt}, src_idx_{src_idx}, format_{format} {}

void updater::reset_vdv_run_ids_() { matches_.clear(); }

statistics const& updater::get_cumulative_stats() const {
  return cumulative_stats_;
}

source_idx_t updater::get_src() const { return src_idx_; }
updater::xml_format updater::get_format() const { return format_; }

template <typename... Formats>
std::optional<unixtime_t> get_opt_time(pugi::xml_node const& node,
                                       char const* str,
                                       Formats... formats) {
  auto const xpath = node.select_node(str);
  if (xpath != nullptr) {
    try {
      return parse_time(xpath.node().child_value(), formats...);
    } catch (std::exception const& e) {
      log(log_lvl::error, "vdv_update.get_opt_time",
          "{}, invalid time input: {}", e.what(), xpath.node().child_value());
    }
  }
  return std::nullopt;
}

std::optional<bool> get_opt_bool(
    pugi::xml_node const& node,
    char const* key,
    std::optional<bool> default_to = std::nullopt) {
  auto const xpath = node.select_node(key);
  return xpath != nullptr ? utl::parse<bool>(xpath.node().child_value())
                          : default_to;
}

std::optional<std::string_view> get_opt_str(
    pugi::xml_node const& node,
    char const* key,
    std::optional<std::string_view> default_to = std::nullopt) {
  auto const xpath = node.select_node(key);
  return xpath != nullptr ? xpath.node().child_value() : default_to;
}

bool is_vdv(updater::xml_format const f) {
  return f == updater::xml_format::kVdv;
}

updater::vdv_stop::vdv_stop(nigiri::location_idx_t const l,
                            std::string_view id,
                            pugi::xml_node const n,
                            xml_format const f)
    : l_{l},
      id_{id},
      dep_{is_vdv(f)
               ? get_opt_time(n, "Abfahrtszeit", "%FT%T")
               : get_opt_time(n, "AimedDepartureTime", "%FT%T%Ez", "%FT%TZ")},
      arr_{is_vdv(f)
               ? get_opt_time(n, "Ankunftszeit", "%FT%T")
               : get_opt_time(n, "AimedArrivalTime", "%FT%T%Ez", "%FT%TZ")},
      rt_dep_{
          is_vdv(f)
              ? get_opt_time(n, "IstAbfahrtPrognose", "%FT%T")
              : get_opt_time(n, "ExpectedDepartureTime", "%FT%T%Ez", "%FT%TZ")},
      rt_arr_{is_vdv(f) ? get_opt_time(n, "IstAnkunftPrognose", "%FT%T")
                        : get_opt_time(
                              n, "ExpectedArrivalTime", "%FT%T%Ez", "%FT%TZ")},
      in_forbidden_{is_vdv(f) ? *get_opt_bool(n, "Einsteigeverbot", false)
                              : *get_opt_str(n,
                                             "DepartureBoardingActivity",
                                             "boarding") != "boarding"},
      out_forbidden_{is_vdv(f) ? *get_opt_bool(n, "Aussteigeverbot", false)
                               : get_opt_str(n,
                                             "ArrivalBoardingActivity",
                                             "alighting") != "alighting"},
      passing_through_{is_vdv(f) ? *get_opt_bool(n, "Durchfahrt", false)
                                 : in_forbidden_ && out_forbidden_},
      arr_canceled_{is_vdv(f) ? *get_opt_bool(n, "AnkunftFaelltAus", false)
                              : *get_opt_bool(n, "Cancellation", false)},
      dep_canceled_{is_vdv(f) ? *get_opt_bool(n, "AbfahrtFaelltAus", false)
                              : arr_canceled_} {}

std::optional<std::pair<unixtime_t, event_type>> updater::vdv_stop::get_event(
    std::optional<event_type> const et) const {
  if ((!et || et == event_type::kDep) && dep_) {
    return std::pair{*dep_, event_type::kDep};
  } else if ((!et || et == event_type::kArr) && arr_) {
    return std::pair{*arr_, event_type::kArr};
  } else {
    return std::nullopt;
  }
}

pugi::xml_node get(pugi::xml_node const& node, char const* str) {
  auto const xpath = node.select_node(str);
  utl::verify(xpath != nullptr, "required xml node not found: {}", str);
  return xpath.node();
}

vector<updater::vdv_stop> updater::resolve_stops(pugi::xml_node const vdv_run,
                                                 statistics& stats) {
  auto vdv_stops = vector<vdv_stop>{};

  auto const vdv_stop_selectors = {"IstHalt"};
  auto const siri_stop_stop_selectors = {"RecordedCalls/RecordedCall",
                                         "EstimatedCalls/EstimatedCall"};
  for (auto const stop_selector :
       is_vdv(format_) ? vdv_stop_selectors : siri_stop_stop_selectors) {
    for (auto const stop : vdv_run.select_nodes(stop_selector)) {
      ++stats.total_stops_;

      auto const vdv_stop_id = std::string_view{
          get(stop.node(), is_vdv(format_) ? "HaltID" : "StopPointRef")
              .child_value()};
      auto const l = [&]() {
        auto const x = tt_.locations_.find({vdv_stop_id, src_idx_});
        if (x.has_value()) {
          return x;
        } else if (auto const underscore_pos = vdv_stop_id.find('_');
                   underscore_pos != std::string_view::npos) {
          // Extra matching code for VRR SIRI. Remove after data is fixed.
          return tt_.locations_.find(
              {vdv_stop_id.substr(0, underscore_pos + 1U), src_idx_});
        } else {
          return x;
        }
      }();

      if (get_opt_bool(stop.node(),
                       is_vdv(format_) ? "Zusatzhalt" : "ExtraCall", false)
              .value()) {
        ++stats.unsupported_additional_stops_;
        vdv_trace("unsupported additional stop: [id: {}, name: {}]\n",
                  vdv_stop_id, l.has_value() ? l->name_ : "unresolvable");
      }

      if (l.has_value()) {
        ++stats.resolved_stops_;
        vdv_stops.emplace_back(l->l_, vdv_stop_id, stop.node(), format_);
      } else {
        vdv_stops.emplace_back(location_idx_t::invalid(), vdv_stop_id,
                               stop.node(), format_);
        vdv_trace("unresolvable stop: {}\n", vdv_stop_id);
      }
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

void updater::match_run(std::string_view vdv_run_id,
                        vector<vdv_stop> const& vdv_stops,
                        statistics& stats,
                        bool const is_complete_run) {
  ++stats.match_attempts_;

  matches_[vdv_run_id] = match{};
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
          if (stop{s}.location_idx() != l) {
            continue;
          }
          auto const vdv_ev = stop_idx == 0
                                  ? vdv_stop.get_event(event_type::kDep)
                              : stop_idx == location_seq.size() - 1
                                  ? vdv_stop.get_event(event_type::kArr)
                                  : vdv_stop.get_event();
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
                    day_idx_t{nigiri_ev_time.days() - day_shift.count()}};

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
      ++stats.no_transport_found_at_stop_;
    }
    for (auto& c : candidates) {
      c.finish_stop();
    }
  }

  utl::sort(candidates);

  auto const is_match = [&](auto const& c) {
    return c.score_ > candidates.front().score_ * kAdditionalMatchThreshold;
  };

  auto const match_threshold =
      is_complete_run ? kFirstMatchThreshold : kFirstMatchThresholdIncomplete;
  if (!candidates.empty() &&
      (candidates.front().score_ >
       vdv_stops.size() * kExactMatchScore * match_threshold)) {
    for (auto const& c : candidates) {
      if (is_match(c)) {
        matches_[vdv_run_id].runs_.emplace_back(c.r_);
      } else {
        break;
      }
    }
  }

  [[maybe_unused]] auto const candidate_str = [&](candidate const& c) {
    return fmt::format(
        "[line: {}, score: {}, length: {}], dbg: {}",
        tt_.trip_lines_
            [tt_.transport_section_lines_[c.r_.t_.t_idx_].size() == 1
                 ? tt_.transport_section_lines_[c.r_.t_.t_idx_].front()
                 : tt_.transport_section_lines_[c.r_.t_.t_idx_]
                                               [c.r_.stop_range_.from_]]
                .view(),
        c.score_, c.total_length_, tt_.dbg(c.r_.t_.t_idx_));
  };

  if (matches_[vdv_run_id].runs_.empty()) {
    vdv_trace("[vdv_aus] no match for {}, best candidate: {}", vdv_run_id,
              candidates.empty() ? "none" : candidate_str(candidates.front()));
  } else {
    ++stats.matched_runs_;
    if (matches_[vdv_run_id].runs_.size() > 1) {
      ++stats.multiple_matches_;
      vdv_trace("[vdv_aus] multiple matches for {}:", vdv_run_id);
      for (auto const& c : candidates) {
        if (!is_match(c)) {
          break;
        }
        vdv_trace("{}", candidate_str(c));
      }
    }
  }
}

void update_event(rt_timetable& rtt,
                  run_stop const& rs,
                  event_type const et,
                  unixtime_t const new_time,
                  std::optional<duration_t>* delay_propagation = nullptr) {
  auto delay = new_time - rs.scheduled_time(et);
  vdv_trace("update [stop_idx: {}, id: {}, name: {}] {}: {}{}{}\n",
            rs.stop_idx_, rs.id(), rs.name(),
            et == event_type::kArr ? "ARR" : "DEP", rs.scheduled_time(et),
            delay.count() >= 0 ? "+" : "", delay.count());
  rtt.update_time(rs.fr_->rt_, rs.stop_idx_, et, new_time);
  rtt.dispatch_delay(*rs.fr_, rs.stop_idx_, et, delay);
  if (delay_propagation != nullptr) {
    *delay_propagation = delay;
  }
}

void monotonize(frun& fr, rt_timetable& rtt) {
  vdv_trace("---monotonizing {}, stop_range: [{}, {}[\n", fr.name(),
            fr.stop_range_.from_, fr.stop_range_.to_);

  auto upper_bound = unixtime_t::max();
  for (auto i = stop_idx_t{0U}; i != fr.size(); ++i) {
    auto const rs = run_stop{
        &fr,
        static_cast<stop_idx_t>(static_cast<stop_idx_t>(fr.size()) - 1U - i)};
    if (rs.stop_idx_ != fr.size() - 1) {
      upper_bound = std::min(rs.time(event_type::kDep), upper_bound);
      update_event(rtt, rs, event_type::kDep, upper_bound);
    }
    if (rs.stop_idx_ != 0) {
      upper_bound = std::min(rs.time(event_type::kArr), upper_bound);
      update_event(rtt, rs, event_type::kArr, upper_bound);
    }
  }
}

void handle_first_last_cancelation(frun& fr, rt_timetable& rtt) {
  auto const cancel_stop = [&](auto& rs) {
    auto& stp = rtt.rt_transport_location_seq_[fr.rt_][rs.stop_idx_];
    stp = stop{stop{stp}.location_idx(), false, false, false, false}.value();
  };

  auto first = fr[0U];
  if (!first.in_allowed()) {
    cancel_stop(first);
  }

  auto last = fr[static_cast<stop_idx_t>(fr.stop_range_.size()) - 1U];
  if (!last.out_allowed()) {
    cancel_stop(last);
  }
}

void updater::update_run(rt_timetable& rtt,
                         run const& r,
                         vector<vdv_stop> const& vdv_stops,
                         bool const is_complete_run,
                         statistics& stats) {
  auto fr = rt::frun(tt_, &rtt, r);
  if (!fr.is_rt()) {
    fr.rt_ = rtt.add_rt_transport(src_idx_, tt_, fr.t_);
  } else {
    rtt.rt_transport_is_cancelled_.set(to_idx(fr.rt_), false);
  }

  auto delay = std::optional<duration_t>{};

  auto const propagate_delay = [&](auto const& rs, event_type et) {
    vdv_trace("propagate [stop_idx: {}, id: {}, name: {}] {}: {}{}{}\n",
              rs.stop_idx_, rs.id(), rs.name(),
              et == event_type::kArr ? "ARR" : "DEP", rs.scheduled_time(et),
              delay->count() >= 0 ? "+" : "", delay->count());
    rtt.update_time(fr.rt_, rs.stop_idx_, et, rs.scheduled_time(et) + *delay);
    rtt.dispatch_delay(fr, rs.stop_idx_, et, *delay);
    ++stats.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto skipped_stops = std::vector<vdv_stop>{};
  auto const print_skipped_stops = [&]() {
    for (auto const& s [[maybe_unused]] : skipped_stops) {
      ++stats.skipped_vdv_stops_;
      vdv_trace("skipped vdv stop: [id: {}, name: {}]\n", s.id_,
                s.l_ == location_idx_t::invalid()
                    ? "unresolvable"
                    : tt_.locations_.get(s.l_).name_);
    }
  };

  vdv_trace("---updating {}, stop_range: [{}, {}[\n", fr.name(),
            fr.stop_range_.from_, fr.stop_range_.to_);
  for (auto i = stop_idx_t{0U}; i != fr.stop_range_.size(); ++i) {
    auto const rs = fr[i];
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
            ++stats.updated_events_;
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
            ++stats.updated_events_;
          }
        }

        if (matched_arr || matched_dep) {  // stop change
          auto& stp = rtt.rt_transport_location_seq_[fr.rt_][rs.stop_idx_];
          auto const in_allowed_update = !vdv_stop->in_forbidden_ &&
                                         !vdv_stop->passing_through_ &&
                                         !vdv_stop->dep_canceled_;
          auto const out_allowed_update = !vdv_stop->out_forbidden_ &&
                                          !vdv_stop->passing_through_ &&
                                          !vdv_stop->arr_canceled_;

          stp = stop{stop{stp}.location_idx(), in_allowed_update,
                     out_allowed_update,
                     rs.get_scheduled_stop().in_allowed_wheelchair() &&
                         in_allowed_update,
                     rs.get_scheduled_stop().out_allowed_wheelchair() &&
                         out_allowed_update}
                    .value();

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
    ++stats.excess_vdv_stops_;
    ++cursor;
  }

  handle_first_last_cancelation(fr, rtt);
  auto const n_not_cancelled_stops = utl::count_if(
      rtt.rt_transport_location_seq_[fr.rt_],
      [](stop::value_type const s) { return !stop{s}.is_cancelled(); });
  if (n_not_cancelled_stops <= 1U) {
    rtt.cancel_run(fr);
  }

  if (!fr.is_cancelled()) {
    monotonize(fr, rtt);
  }
}

void updater::process_vdv_run(rt_timetable& rtt,
                              pugi::xml_node const vdv_run,
                              statistics& stats) {
  ++stats.total_runs_;
  auto const is_complete_run = *get_opt_bool(
      vdv_run,
      format_ == xml_format::kVdv ? "Komplettfahrt" : "IsCompleteStopSequence",
      false);
  if (is_complete_run) {
    ++stats.complete_runs_;
  }

  auto vdv_stops = resolve_stops(vdv_run, stats);

  auto const run_ref_selector =
      format_ == xml_format::kVdv ? "./FahrtRef/FahrtID/FahrtBezeichner"
      : format_ == xml_format::kSiri
          ? "./FramedVehicleJourneyRef/DatedVehicleJourneyRef"
          : "./DatedVehicleJourneyRef";
  auto const operation_day_selector =
      format_ == xml_format::kVdv    ? "./FahrtRef/FahrtID/Betriebstag"
      : format_ == xml_format::kSiri ? "./FramedVehicleJourneyRef/DataFrameRef"
                                     : "./DatedVehicleJourneyRef";
  auto const vdv_run_id =
      fmt::format("{}{}", get(vdv_run, run_ref_selector).child_value(),
                  get(vdv_run, operation_day_selector).child_value());

  if (vdv_stops.empty()) {
    ++stats.runs_without_stops_;
    vdv_trace("vdv run without stops: {}\n", vdv_run_id);
    return;
  }

  auto const seen_before = matches_.contains(vdv_run_id);
  if (!seen_before) {
    ++stats.unique_runs_;
    if (is_complete_run) {
      match_run(vdv_run_id, vdv_stops, stats, is_complete_run);
    } else {
      ++stats.incomplete_not_seen_before_;
      match_run(vdv_run_id, vdv_stops, stats, is_complete_run);
      matches_[vdv_run_id].only_saw_incomplete_ = true;
    }
  }

  if (seen_before && is_complete_run &&
      matches_[vdv_run_id].only_saw_incomplete_) {
    ++stats.complete_after_incomplete_;
    match_run(vdv_run_id, vdv_stops, stats, is_complete_run);
    matches_[vdv_run_id].only_saw_incomplete_ = false;
  }

  auto const& runs = matches_[vdv_run_id].runs_;
  for (auto& r : runs) {
    auto const cancelled_run_selector =
        format_ == xml_format::kVdv ? "FaelltAus" : "Cancellation";
    if (get_opt_bool(vdv_run, cancelled_run_selector, false).value()) {
      rtt.cancel_run(r);
      ++stats.cancelled_runs_;
    } else {
      update_run(rtt, r, vdv_stops, is_complete_run, stats);
    }
  }

  stats.found_runs_ += runs.empty() ? 0U : 1U;

  matches_[vdv_run_id].last_accessed_ =
      std::chrono::time_point_cast<std::chrono::seconds>(
          std::chrono::system_clock::now());
}

void updater::clean_up() {
  auto const now = std::chrono::time_point_cast<std::chrono::seconds>(
      std::chrono::system_clock::now());
  std::erase_if(matches_, [&](auto const& i) {
    auto const& [vdv_id, match] = i;
    return now - match.last_accessed_ > kMatchRetention;
  });
  last_cleanup = now;
}

statistics updater::update(rt_timetable& rtt, pugi::xml_document const& doc) {
  if (std::chrono::system_clock::now() - last_cleanup > kCleanUpInterval) {
    clean_up();
  }

  auto stats = statistics{};
  for (auto const& vdv_run : doc.select_nodes(
           is_vdv(format_) ? "//IstFahrt" : "//EstimatedVehicleJourney")) {
    if (*get_opt_bool(vdv_run.node(),
                      is_vdv(format_) ? "Zusatzfahrt" : "ExtraJourney",
                      false)) {
#ifdef VDV_DEBUG
      vdv_trace("unsupported additional run:\n");
      vdv_run.node().print(std::cout);
#endif
      ++stats.unsupported_additional_runs_;
      continue;
    }
    process_vdv_run(rtt, vdv_run.node(), stats);
  }

  cumulative_stats_ += stats;
  cumulative_stats_.current_matches_total_ = matches_.size();
  cumulative_stats_.current_matches_non_empty_ = [&]() {
    auto n = 0U;
    for (auto const& [_, m] : matches_) {
      n += m.runs_.empty() ? 0U : 1U;
    }
    return n;
  }();

  return stats;
}

}  // namespace nigiri::rt::vdv_aus
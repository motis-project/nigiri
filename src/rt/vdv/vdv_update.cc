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

template <typename T>
bool epsilon_equal(T const& a, T const& b) {
  constexpr auto const epsilon = T{1};
  return a - epsilon <= b && b <= a + epsilon;
}

bool epsilon_equal_unixtime(unixtime_t const a, unixtime_t const b) {
  return epsilon_equal(a.time_since_epoch().count(),
                       b.time_since_epoch().count());
}

vector<updater::vdv_stop> updater::resolve_stops(pugi::xml_node const vdv_run) {
  auto vdv_stops = vector<vdv_stop>{};

  auto additional_stops = std::ofstream{"additional_stops.txt", std::ios::app};

  auto unresolvable_stops =
      std::ofstream{"unresolvable_stops.txt", std::ios::app};

  for (auto const stop : vdv_run.select_nodes("IstHalt")) {
    ++stats_.total_stops_;

    auto const vdv_stop_id =
        std::string_view{get(stop.node(), "HaltID").child_value()};
    auto const l = tt_.locations_.find({vdv_stop_id, src_idx_});

    if (get_opt_bool(stop.node(), "Zusatzhalt", false).value()) {
      ++stats_.unsupported_additional_stops_;
      additional_stops << "[id: " << vdv_stop_id << ", name:"
                       << (l.has_value() ? l->name_ : "unresolvable") << "]\n";
    }

    if (l.has_value()) {
      ++stats_.resolved_stops_;
      vdv_stops.emplace_back(l->l_, vdv_stop_id, stop.node());
    } else {
      ++stats_.unknown_stops_;
      unresolvable_stops << vdv_stop_id << "\n";
      vdv_stops.emplace_back(location_idx_t::invalid(), vdv_stop_id,
                             stop.node());
    }
  }

  return vdv_stops;
}

std::optional<rt::run> updater::find_run(pugi::xml_node const vdv_run,
                                         std::string_view vdv_run_id,
                                         vector<vdv_stop> const& vdv_stops,
                                         bool const is_complete_run) {
  if (!is_complete_run) {
    ++stats_.search_on_incomplete_;
    log(log_lvl::error, "vdv_updater.find_run",
        "Warning: attempting to match an incomplete VDV run");
  }

  struct candidate {
    run r_;
    std::uint32_t n_matches_;
  };

  auto candidates = std::vector<candidate>{};

  for (auto const& vdv_stop : vdv_stops) {
    if (vdv_stop.l_ == location_idx_t::invalid()) {
      continue;
    }
    auto no_transport_found_at_stop = true;
    for (auto const r : tt_.location_routes_[vdv_stop.l_]) {
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
          if (!epsilon_equal(ev_time.mam(), mam.count())) {
            continue;
          }

          auto const tr = transport{tt_.route_transport_ranges_[r][ev_time_idx],
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
                  0U);
              candidate = end(candidates) - 1;
            }
            ++candidate->n_matches_;
            no_transport_found_at_stop = false;
          }
        }
      }
    }
    if (no_transport_found_at_stop) {
      ++stats_.no_transport_found_at_stop_;
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::sort(
      begin(candidates), end(candidates),
      [](auto const& a, auto const& b) { return a.n_matches_ > b.n_matches_; });

  if (candidates.size() > 1) {
    if (candidates[0].n_matches_ == candidates[1].n_matches_) {
      ++stats_.multiple_matches_;
      auto multiple_matches =
          std::ofstream{"multiple_matches.txt", std::ios::app};
      multiple_matches << "multiple match candidates:\n";
      for (auto const& c : candidates) {
        multiple_matches
            << "[line: "
            << tt_.trip_lines_
                   [tt_.transport_section_lines_[c.r_.t_.t_idx_].size() == 1
                        ? tt_.transport_section_lines_[c.r_.t_.t_idx_].front()
                        : tt_.transport_section_lines_[c.r_.t_.t_idx_]
                                                      [c.r_.stop_range_.from_]]
                       .view()
            << ", #matching_stops: " << c.n_matches_ << "]\n";
      }
      multiple_matches << "for update:\n";
      vdv_run.print(multiple_matches);
      multiple_matches << "\n";
    }
  }

  ++stats_.found_runs_;
  vdv_nigiri_[vdv_run_id] = candidates.front().r_;
  return candidates.front().r_;
}

void updater::update_run(rt_timetable& rtt,
                         run& r,
                         vector<vdv_stop> const& vdv_stops,
                         bool const is_complete_run) {
  auto fr = rt::frun(tt_, &rtt, r);
  if (!fr.is_rt()) {
    fr.rt_ = rtt.add_rt_transport(src_idx_, tt_, r.t_);
  }

  auto update_events = std::ofstream{"update_events.txt", std::ios::app};

  update_events << "---updating " << fr.name()
                << ", stop_idx: " << fr.stop_range_.from_ << " to "
                << fr.stop_range_.to_ - 1 << "\n\n";

  auto gtfs_stop_missing = std::stringstream{};
  auto prefix_matches = std::stringstream{};

  auto delay = std::optional<duration_t>{};

  auto const update_event = [&](auto const& rs, auto const et,
                                auto const new_time) {
    delay = new_time - rs.scheduled_time(et);
    update_events << "update " << (et == event_type::kArr ? "ARR: " : "DEP: ")
                  << rs.scheduled_time(et) << "+" << delay->count() << "\n";
    rtt.update_time(fr.rt_, rs.stop_idx_, et, new_time);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats_.updated_events_;
  };

  auto const propagate_delay = [&](auto const& rs, event_type et) {
    update_events << rs.scheduled_time(et) << "+" << delay->count() << "\n";
    rtt.update_time(fr.rt_, rs.stop_idx_, et, rs.scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats_.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto skipped_stops = std::vector<vdv_stop>{};
  auto const print_skipped_stops = [&]() {
    for (auto const& s : skipped_stops) {
      ++stats_.skipped_vdv_stops_;
      update_events << "skipped vdv stop: [id: " << s.id_ << ", name: "
                    << (s.l_ == location_idx_t::invalid()
                            ? "unresolvable"
                            : tt_.locations_.get(s.l_).name_)
                    << "]\n";
    }
  };

  auto const prefix_match = [](auto const& a, auto const& b) {
    auto colons = 0U;
    for (auto const& [el_a, el_b] : std::views::zip(a, b)) {
      if (el_a != el_b) {
        return false;
      }
      if (el_a == ':') {
        ++colons;
      }
      if (colons == 3U) {
        break;
      }
    }
    return true;
  };

  for (auto const rs : fr) {
    auto matched_arr = false;
    auto matched_dep = false;
    skipped_stops.clear();
    for (auto vdv_stop = cursor; vdv_stop != end(vdv_stops); ++vdv_stop) {
      if (vdv_stop->l_ != location_idx_t::invalid() &&
          matches(tt_, routing::location_match_mode::kEquivalent,
                  rs.get_location_idx(), vdv_stop->l_)) {
        update_events << "location match at stop_idx = " << rs.stop_idx_
                      << ": [id: " << rs.id() << ", name: " << rs.name()
                      << "]\n";
        if (rs.stop_idx_ != 0 && vdv_stop->arr_.has_value() &&
            epsilon_equal_unixtime(vdv_stop->arr_.value(),
                                   rs.scheduled_time(event_type::kArr))) {
          matched_arr = true;
          if (vdv_stop->rt_arr_.has_value()) {
            update_event(rs, event_type::kArr, *vdv_stop->rt_arr_);
          }
        }
        if (rs.stop_idx_ != fr.stop_range_.to_ - 1 &&
            vdv_stop->dep_.has_value() &&
            epsilon_equal_unixtime(vdv_stop->dep_.value(),
                                   rs.scheduled_time(event_type::kDep))) {
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
      } else if (prefix_match(vdv_stop->id_, rs.id())) {
        prefix_matches << "vdv_stop_idx = "
                       << std::distance(begin(vdv_stops), vdv_stop) << ": "
                       << vdv_stop->id_ << "\ngtfs_stop_idx = " << rs.stop_idx_
                       << ": " << rs.id() << "\n\n";
      }
      skipped_stops.emplace_back(*vdv_stop);
    }
    if (!matched_arr && !matched_dep && is_complete_run &&
        rs.stop_idx_ < vdv_stops.size()) {
      gtfs_stop_missing << "stop_idx = " << rs.stop_idx_ << ": [id: " << rs.id()
                        << ", name: " << rs.name() << "]\n";
    }
    if (delay) {
      if (rs.stop_idx_ != 0 && !matched_arr) {
        update_events << "propagating delay at stop_idx = " << rs.stop_idx_
                      << ": [id: " << rs.id() << ", name: " << rs.name()
                      << "] ARR: ";
        propagate_delay(rs, event_type::kArr);
      }
      if (rs.stop_idx_ != fr.stop_range_.to_ - 1 && !matched_dep) {
        update_events << "propagating delay at stop_idx = " << rs.stop_idx_
                      << ": [id: " << rs.id() << ", name: " << rs.name()
                      << "] DEP: ";
        propagate_delay(rs, event_type::kDep);
      }
    }
    update_events << "\n";
  }

  while (cursor != end(vdv_stops)) {
    update_events << "excess vdv stop: [id: " << cursor->id_ << ", name: "
                  << (cursor->l_ == location_idx_t::invalid()
                          ? "unresolvable"
                          : tt_.locations_.get(cursor->l_).name_)
                  << "]\n";
    ++stats_.excess_vdv_stops_;
    ++cursor;
  }

  if (!gtfs_stop_missing.str().empty()) {
    std::ofstream{"gtfs_stop_missing_in_vdv.txt", std::ios::app}
        << "---" << fr.name() << ":\n"
        << gtfs_stop_missing.str() << "---\n\n";
  }

  if (!prefix_matches.str().empty()) {
    std::ofstream{"prefix_only_matches.txt", std::ios::app}
        << "---" << fr.name() << ":\n"
        << prefix_matches.str() << "---\n\n";
  }
  update_events << "---\n\n";
}

void updater::process_vdv_run(rt_timetable& rtt, pugi::xml_node const vdv_run) {
  ++stats_.total_runs_;

  auto vdv_stops = resolve_stops(vdv_run);

  if (vdv_stops.empty()) {
    ++stats_.runs_without_stops_;
    auto run_without_stops =
        std::ofstream{"runs_without_stops.txt", std::ios::app};
    vdv_run.print(run_without_stops);
    run_without_stops << "\n";
    return;
  }

  auto const is_complete_run = *get_opt_bool(vdv_run, "Komplettfahrt", false);

  auto const vdv_run_id =
      std::string{
          get(vdv_run, "./FahrtRef/FahrtID/FahrtBezeichner").child_value()}
          .append(get(vdv_run, "./FahrtRef/FahrtID/Betriebstag").child_value());

  auto r = vdv_nigiri_.contains(vdv_run_id)
               ? std::optional{vdv_nigiri_.at(vdv_run_id)}
               : find_run(vdv_run, vdv_run_id, vdv_stops, is_complete_run);
  if (!r.has_value()) {
    ++stats_.unmatchable_runs_;
    auto unmatchable_runs =
        std::ofstream{"unmatchable_runs.txt", std::ios::app};
    vdv_run.print(unmatchable_runs);
    unmatchable_runs << "\n";
    return;
  }
  ++stats_.matched_runs_;

  update_run(rtt, *r, vdv_stops, is_complete_run);
}

void updater::update(rt_timetable& rtt, pugi::xml_document const& doc) {
  auto stats = statistics{};
  for (auto const& vdv_run : doc.select_nodes("//IstFahrt")) {
    if (get_opt_bool(vdv_run.node(), "Zusatzfahrt", false).value()) {
      ++stats.unsupported_additional_runs_;
      auto additional_runs =
          std::ofstream{"additional_runs.txt", std::ios::app};
      vdv_run.node().print(additional_runs);
      additional_runs << "\n";
      continue;
    } else if (get_opt_bool(vdv_run.node(), "FaelltAus", false).value()) {
      ++stats.unsupported_cancelled_runs_;
      auto canceled_runs = std::ofstream{"canceled_runs.txt", std::ios::app};
      vdv_run.node().print(canceled_runs);
      canceled_runs << "\n";
      continue;
    }

    process_vdv_run(rtt, vdv_run.node());
  }
}

}  // namespace nigiri::rt::vdv
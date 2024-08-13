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
  explicit vdv_stop(location_idx_t const l,
                    std::string_view id,
                    pugi::xml_node const n)
      : l_{l},
        id_{id},
        dep_{get_opt_time(n, "Abfahrtszeit")},
        arr_{get_opt_time(n, "Ankunftszeit")},
        rt_dep_{get_opt_time(n, "IstAbfahrtPrognose")},
        rt_arr_{get_opt_time(n, "IstAnkunftPrognose")} {}

  std::optional<std::pair<unixtime_t, event_type>> get_event(
      event_type et) const {
    if (et == event_type::kArr && arr_.has_value()) {
      return std::pair{*arr_, event_type::kArr};
    } else if (et == event_type::kDep && dep_.has_value()) {
      return std::pair{*dep_, event_type::kDep};
    } else {
      return std::nullopt;
    }
  }

  location_idx_t l_;
  std::string_view id_;
  std::optional<unixtime_t> dep_, arr_, rt_dep_, rt_arr_;
};

vector<vdv_stop> resolve_stops(timetable const& tt,
                               source_idx_t const src,
                               pugi::xml_node const run,
                               statistics& stats) {
  auto vdv_stops = vector<vdv_stop>{};

  auto additional_stops = std::ofstream{"additional_stops.txt", std::ios::app};

  auto unresolvable_stops =
      std::ofstream{"unresolvable_stops.txt", std::ios::app};

  for (auto const stop : run.select_nodes("IstHalt")) {
    ++stats.total_stops_;

    auto const vdv_stop_id =
        std::string_view{get(stop.node(), "HaltID").child_value()};
    auto const l = tt.locations_.find({vdv_stop_id, src});

    if (get_opt_bool(stop.node(), "Zusatzhalt", false).value()) {
      ++stats.unsupported_additional_stops_;
      additional_stops << "[id: " << vdv_stop_id << ", name:"
                       << (l.has_value() ? l->name_ : "unresolvable") << "]\n";
      continue;
    }

    if (l.has_value()) {
      ++stats.resolved_stops_;
      vdv_stops.emplace_back(l->l_, vdv_stop_id, stop.node());
    } else {
      ++stats.unknown_stops_;
      unresolvable_stops << vdv_stop_id << "\n";
      vdv_stops.emplace_back(location_idx_t::invalid(), vdv_stop_id,
                             stop.node());
    }
  }

  return vdv_stops;
}

std::optional<rt::run> find_run(timetable const& tt,
                                pugi::xml_node const run,
                                auto const& vdv_stops,
                                statistics& stats) {
  using namespace std::literals;

  auto const vdv_line_text_xpath = run.select_node("LinienText");
  if (!vdv_line_text_xpath) {
    std::cout << "VDV run without line text:\n";
    return std::nullopt;
  }
  auto vdv_line_text = std::string{vdv_line_text_xpath.node().child_value()};
  std::erase_if(vdv_line_text, [](auto const c) { return c == ' '; });

  auto const vdv_direction_id_xpath = run.select_node("RichtungsID");
  if (!vdv_direction_id_xpath) {
    std::cout << "VDV run without direction id:\n";
    return std::nullopt;
  }

  auto candidates =
      hash_map<transport, std::pair<interval<stop_idx_t>, unsigned>>{};

  for (auto const& vdv_stop : vdv_stops) {
    if (vdv_stop.l_ == location_idx_t::invalid()) {
      continue;
    }
    auto no_transport_found_at_stop = true;
    for (auto const r : tt.location_routes_[vdv_stop.l_]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
        if (!matches(tt, routing::location_match_mode::kEquivalent,
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
        auto const [day_idx, mam] = tt.day_idx_mam(t);

        for (auto const& [ev_time_idx, ev_time] :
             utl::enumerate(tt.event_times_at_stop(
                 r, static_cast<stop_idx_t>(stop_idx), ev_type))) {
          if (ev_time.mam() != mam.count()) {
            continue;
          }

          auto const tr = transport{tt.route_transport_ranges_[r][ev_time_idx],
                                    day_idx - day_idx_t{ev_time.days()}};

          if (tt.bitfields_[tt.transport_traffic_days_[tr.t_idx_]].test(
                  to_idx(tr.day_))) {

            auto const trip_line =
                size(tt.transport_section_lines_[tr.t_idx_]) == 1U
                    ? tt.trip_lines_[tt.transport_section_lines_[tr.t_idx_][0U]]
                          .view()
                    : tt.trip_lines_[tt.transport_section_lines_[tr.t_idx_]
                                                                [stop_idx]]
                          .view();

            if (vdv_line_text.find(trip_line.substr(0, trip_line.find(' '))) ==
                std::string_view::npos) {
              ++stats.match_prevented_by_line_id_;
              continue;
            }

            if (!candidates.contains(tr)) {
              candidates[tr] = {{static_cast<stop_idx_t>(stop_idx),
                                 static_cast<stop_idx_t>(location_seq.size())},
                                0U};
            }

            if (++candidates[tr].second == 10) {
              return rt::run{
                  tr, {candidates[tr].first.from_, candidates[tr].first.to_}};
            }

            no_transport_found_at_stop = false;
          }
        }
      }
    }
    if (no_transport_found_at_stop) {
      ++stats.no_transport_found_at_stop_;
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }

  std::cout << "match candidates:\n";
  for (auto const& [k, v] : candidates) {
    std::cout
        << "[line: "
        << tt.trip_lines_[tt.transport_section_lines_[k.t_idx_].size() == 1
                              ? tt.transport_section_lines_[k.t_idx_].front()
                              : tt.transport_section_lines_
                                    [k.t_idx_][candidates[k].first.from_]]
               .view()
        << ", #matching_stops: " << v.second << "]\n";
  }

  auto const most_matches = std::max_element(
      begin(candidates), end(candidates), [](auto const& a, auto const& b) {
        return a.second.second < b.second.second;
      });

  return rt::run{
      most_matches->first,
      {most_matches->second.first.from_, most_matches->second.first.to_}};
}

void update_run(timetable const& tt,
                rt_timetable& rtt,
                source_idx_t const src,
                run& r,
                auto const& vdv_stops,
                bool const is_complete_run,
                statistics& stats) {
  auto fr = rt::frun(tt, &rtt, r);
  if (!fr.is_rt()) {
    fr.rt_ = rtt.add_rt_transport(src, tt, r.t_);
  }

  std::cout << "---updating " << fr.name()
            << ", stop_idx: " << fr.stop_range_.from_ << " to "
            << fr.stop_range_.to_ - 1 << "\n\n";

  auto gtfs_stop_missing = std::stringstream{};
  auto prefix_matches = std::stringstream{};

  auto delay = std::optional<duration_t>{};

  auto const update_event = [&](auto const& rs, auto const et,
                                auto const new_time) {
    delay = new_time - rs.scheduled_time(et);
    std::cout << "update " << (et == event_type::kArr ? "ARR: " : "DEP: ")
              << rs.scheduled_time(et) << "+" << delay->count() << "\n";
    rtt.update_time(fr.rt_, rs.stop_idx_, et, new_time);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats.updated_events_;
  };

  auto const propagate_delay = [&](auto const& rs, event_type et) {
    std::cout << rs.scheduled_time(et) << "+" << delay->count() << "\n";
    rtt.update_time(fr.rt_, rs.stop_idx_, et, rs.scheduled_time(et) + *delay);
    rtt.dispatch_event_change(fr.t_, rs.stop_idx_, et, *delay, false);
    ++stats.propagated_delays_;
  };

  auto cursor = begin(vdv_stops);
  auto skipped_stops = std::vector<vdv_stop>{};
  auto const print_skipped_stops = [&]() {
    for (auto const& s : skipped_stops) {
      ++stats.skipped_vdv_stops_;
      std::cout << "skipped vdv stop: [id: " << s.id_ << ", name: "
                << (s.l_ == location_idx_t::invalid()
                        ? "unresolvable"
                        : tt.locations_.get(s.l_).name_)
                << "]\n";
    }
    std::cout << "\n";
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
          matches(tt, routing::location_match_mode::kEquivalent,
                  rs.get_location_idx(), vdv_stop->l_)) {
        std::cout << "location match at stop_idx = " << rs.stop_idx_
                  << ": [id: " << rs.id() << ", name: " << rs.name() << "]\n";
        if (rs.stop_idx_ != 0 && vdv_stop->arr_.has_value() &&
            vdv_stop->arr_.value() == rs.scheduled_time(event_type::kArr)) {
          matched_arr = true;
          if (vdv_stop->rt_arr_.has_value()) {
            update_event(rs, event_type::kArr, *vdv_stop->rt_arr_);
          }
        }
        if (rs.stop_idx_ != fr.stop_range_.to_ - 1 &&
            vdv_stop->dep_.has_value() &&
            vdv_stop->dep_.value() == rs.scheduled_time(event_type::kDep)) {
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
        std::cout << "propagating delay at stop_idx = " << rs.stop_idx_
                  << ": [id: " << rs.id() << ", name: " << rs.name()
                  << "] ARR: ";
        propagate_delay(rs, event_type::kArr);
      }
      if (rs.stop_idx_ != fr.stop_range_.to_ - 1 && !matched_dep) {
        std::cout << "propagating delay at stop_idx = " << rs.stop_idx_
                  << ": [id: " << rs.id() << ", name: " << rs.name()
                  << "] DEP: ";
        propagate_delay(rs, event_type::kDep);
      }
      std::cout << "\n";
    }
  }

  while (cursor != end(vdv_stops)) {
    std::cout << "excess vdv stop: [id: " << cursor->id_ << ", name: "
              << (cursor->l_ == location_idx_t::invalid()
                      ? "unresolvable"
                      : tt.locations_.get(cursor->l_).name_)
              << "]\n";
    ++stats.excess_vdv_stops_;
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
  std::cout << "---\n\n";
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
    auto unmatchable_runs =
        std::ofstream{"unmatchable_runs.txt", std::ios::app};
    run.print(unmatchable_runs);
    unmatchable_runs << "\n";
    return;
  }
  ++stats.matched_runs_;

  auto const is_complete_run = *get_opt_bool(run, "Komplettfahrt", false);

  update_run(tt, rtt, src, *r, vdv_stops, is_complete_run, stats);
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
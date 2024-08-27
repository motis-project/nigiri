#pragma once

#include "pugixml.hpp"

#include "nigiri/rt/run.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt::vdv {

struct statistics {
  friend std::ostream& operator<<(std::ostream& out, statistics const& s) {
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

  friend statistics& operator+=(statistics& lhs, statistics const& rhs) {
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

  std::uint32_t unsupported_additional_runs_{0U};
  std::uint32_t unsupported_cancelled_runs_{0U};
  std::uint32_t total_stops_{0U};
  std::uint32_t resolved_stops_{0U};
  std::uint32_t unknown_stops_{0U};
  std::uint32_t unsupported_additional_stops_{0U};
  std::uint32_t total_runs_{0U};
  std::uint32_t no_transport_found_at_stop_{0U};
  std::uint32_t search_on_incomplete_{0U};
  std::uint32_t found_runs_{0U};
  std::uint32_t multiple_matches_{0U};
  std::uint32_t matched_runs_{0U};
  std::uint32_t unmatchable_runs_{0U};
  std::uint32_t runs_without_stops_{0U};
  std::uint32_t skipped_vdv_stops_{0U};
  std::uint32_t excess_vdv_stops_{0U};
  std::uint32_t updated_events_{0U};
  std::uint32_t propagated_delays_{0U};
};

struct updater {
  static constexpr auto const kAllowedTimeDiscrepancy = 10U;  // minutes

  updater(timetable const& tt, source_idx_t const src_idx)
      : tt_{tt}, src_idx_{src_idx} {}

  void reset_vdv_run_ids_() { vdv_nigiri_.clear(); }

  statistics const& get_stats() const { return stats_; }

  void update(rt_timetable&, pugi::xml_document const&);

private:
  static std::optional<unixtime_t> get_opt_time(pugi::xml_node const&,
                                                char const*);

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

  vector<vdv_stop> resolve_stops(pugi::xml_node const vdv_run);

  std::optional<rt::run> find_run(std::string_view vdv_run_id,
                                  vector<vdv_stop> const&,
                                  bool is_complete_run);

  void update_run(rt_timetable&,
                  run&,
                  vector<vdv_stop> const&,
                  bool is_complete_run);

  void process_vdv_run(rt_timetable&, pugi::xml_node vdv_run);

  timetable const& tt_;
  source_idx_t src_idx_;
  statistics stats_{};
  hash_map<std::string, run> vdv_nigiri_{};
};

}  // namespace nigiri::rt::vdv
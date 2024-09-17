#pragma once

#include "nigiri/rt/run.h"
#include "nigiri/types.h"

namespace pugi {
struct xml_document;
struct xml_node;
}  // namespace pugi

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt::vdv {

struct statistics {
  friend std::ostream& operator<<(std::ostream&, statistics const&);
  friend statistics& operator+=(statistics&, statistics const&);

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
  static constexpr auto const kExactMatchScore = 1000;
  static constexpr auto const kAllowedTimeDiscrepancy = []() {
    auto error = 0;
    while (kExactMatchScore - error * error > 0) {
      ++error;
    }
    return error - 1;
  }();  // minutes

  updater(timetable const&, source_idx_t);

  void reset_vdv_run_ids_();

  statistics const& get_stats() const;

  void update(rt_timetable&, pugi::xml_document const&);

private:
  static std::optional<unixtime_t> get_opt_time(pugi::xml_node const&,
                                                char const*);

  struct vdv_stop {
    explicit vdv_stop(location_idx_t, std::string_view id, pugi::xml_node);

    std::optional<std::pair<unixtime_t, event_type>> get_event(
        event_type et) const;

    location_idx_t l_;
    std::string_view id_;
    std::optional<unixtime_t> dep_, arr_, rt_dep_, rt_arr_;
  };

  vector<vdv_stop> resolve_stops(pugi::xml_node vdv_run);

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
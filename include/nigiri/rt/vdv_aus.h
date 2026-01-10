#pragma once

#include "nigiri/rt/run.h"
#include "nigiri/rt/service_alert.h"
#include "nigiri/types.h"

#include <optional>
#include <string>

namespace pugi {
class xml_document;
class xml_node;
}  // namespace pugi

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt::vdv_aus {

struct statistics {
  friend std::ostream& operator<<(std::ostream&, statistics const&);
  statistics& operator+=(statistics const&);

  std::uint32_t unsupported_additional_runs_{0U};
  std::uint32_t unsupported_additional_stops_{0U};

  size_t current_matches_total_{0U};
  std::uint32_t current_matches_non_empty_{0U};

  std::uint32_t total_runs_{0U};
  std::uint32_t complete_runs_{0U};
  std::uint32_t unique_runs_{0U};
  std::uint32_t match_attempts_{0U};
  std::uint32_t matched_runs_{0U};
  std::uint32_t found_runs_{0U};
  std::uint32_t multiple_matches_{0U};
  std::uint32_t incomplete_not_seen_before_{0U};
  std::uint32_t complete_after_incomplete_{0U};
  std::uint32_t no_transport_found_at_stop_{0U};

  std::uint32_t total_stops_{0U};
  std::uint32_t resolved_stops_{0U};

  std::uint32_t runs_without_stops_{0U};

  std::uint32_t cancelled_runs_{0U};

  std::uint32_t skipped_vdv_stops_{0U};
  std::uint32_t excess_vdv_stops_{0U};
  std::uint32_t updated_events_{0U};
  std::uint32_t propagated_delays_{0U};

  bool error_{false};
};

struct updater {
  enum class xml_format : std::uint8_t { kVdv, kSiri, kSiriJson, kNumFormats };

  updater(timetable const&, source_idx_t, xml_format format = xml_format::kVdv);

  void reset_vdv_run_ids_();

  statistics const& get_cumulative_stats() const;

  source_idx_t get_src() const;
  xml_format get_format() const;

  statistics update(rt_timetable&, pugi::xml_document const&);

private:
  struct vdv_stop {
    explicit vdv_stop(location_idx_t,
                      std::string_view id,
                      pugi::xml_node,
                      xml_format);

    std::optional<std::pair<unixtime_t, event_type>> get_event(
        std::optional<event_type> et = std::nullopt) const;

    location_idx_t l_;
    std::string_view id_;
    std::optional<unixtime_t> dep_, arr_, rt_dep_, rt_arr_;
    bool in_forbidden_, out_forbidden_, passing_through_, arr_canceled_,
        dep_canceled_;
  };

  struct run_id {
    std::string full_;
    std::string_view run_;
    std::optional<std::string_view> date_{};
  };

  std::optional<run_id> resolve_run_id(pugi::xml_node vdv_run);
  vector<vdv_stop> resolve_stops(pugi::xml_node vdv_run, statistics&);

  void match_run(run_id const&,
                 vector<vdv_stop> const&,
                 statistics&,
                 bool is_complete_run);

  void update_run(rt_timetable&,
                  run const&,
                  vector<vdv_stop> const&,
                  bool is_complete_run,
                  statistics&);

  void affects_alerts(rt_timetable&, pugi::xml_node affects, alert_idx_t);
  void process_vdv_run(rt_timetable&, pugi::xml_node vdv_run, statistics&);
  void process_vdv_alert(rt_timetable&, pugi::xml_node vdv_alert);

  void clean_up();

  struct match {
    std::chrono::sys_seconds last_accessed_{
        std::chrono::time_point_cast<std::chrono::seconds>(
            std::chrono::system_clock::now())};
    bool only_saw_incomplete_{false};
    std::vector<run> runs_{};
  };

  timetable const& tt_;
  source_idx_t src_idx_;
  statistics cumulative_stats_{};
  hash_map<std::string, match> matches_{};
  std::chrono::sys_seconds last_cleanup{
      std::chrono::time_point_cast<std::chrono::seconds>(
          std::chrono::system_clock::now())};
  xml_format format_;
};

}  // namespace nigiri::rt::vdv_aus
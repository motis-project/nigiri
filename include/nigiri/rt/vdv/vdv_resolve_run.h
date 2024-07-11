#pragma once

#include <optional>
#include <string>

#include "date/date.h"
#include "pugixml.hpp"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

struct vdv_stop {
  friend std::ostream& operator<<(std::ostream&, const vdv_stop&);

  std::string stop_id_;
  std::optional<std::string> platform_arr_;
  std::optional<std::string> platform_dep_;
  std::optional<unixtime_t> t_arr_;
  std::optional<unixtime_t> t_dep_;
  std::optional<unixtime_t> t_arr_rt_;
  std::optional<unixtime_t> t_dep_rt_;
  std::optional<bool> in_allowed_;
  std::optional<bool> out_allowed_;
  std::optional<bool> additional_stop_;
};

struct vdv_run {
  unixtime_t t_;
  std::string route_id_;
  std::string route_text_;
  std::string direction_id_;
  std::string direction_text_;
  std::string vehicle_;
  std::string trip_ref_;
  std::string operator_;
  date::sys_days date_;
  bool complete_;  // if false, only stops with updates will be transmitted
  bool canceled_{false};
  bool additional_run_{false};
  std::vector<vdv_stop> stops_;
};

vdv_run parse_run(pugi::xml_node const&);

constexpr auto const kAllowedError = minutes_after_midnight_t::rep{5};

std::optional<location_idx_t> match_location(timetable const&,
                                             std::string_view vdv_stop_id);

template <event_type ET>
void match_time(timetable const&,
                location_idx_t const,
                unixtime_t const,
                hash_set<transport>& matches);

hash_set<transport> match_transport(timetable const&, vdv_run const&);

}  // namespace nigiri::rt
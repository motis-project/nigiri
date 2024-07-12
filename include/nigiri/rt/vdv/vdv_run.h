#pragma once

#include <optional>
#include <string>
#include <vector>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri::rt {

struct vdv_stop {
  std::string stop_id_;
  std::optional<std::string_view> platform_arr_;
  std::optional<std::string_view> platform_dep_;
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
  std::string_view route_id_;
  std::string_view route_text_;
  std::string_view direction_id_;
  std::string_view direction_text_;
  std::string_view vehicle_;
  std::string_view trip_ref_;
  std::string_view operator_;
  date::sys_days date_;
  bool complete_;  // if false, only stops with updates will be transmitted
  bool canceled_{false};
  bool additional_run_{false};
  std::vector<vdv_stop> stops_;
};

}  // namespace nigiri::rt
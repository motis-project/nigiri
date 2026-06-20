#pragma once

#include <filesystem>
#include <vector>

#include "date/date.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
}

namespace nigiri::loader {

struct assistance_times;

struct timetable_source {
  std::string tag_;
  std::string path_;
  loader_config loader_config_{};
};

timetable load(std::vector<timetable_source> const&,
               finalize_options const&,
               interval<date::sys_days> const&,
               assistance_times* = nullptr,
               shapes_storage* = nullptr,
               bool ignore = false);

}  // namespace nigiri::loader
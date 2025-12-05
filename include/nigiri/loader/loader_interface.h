#pragma once

#include <array>
#include <string_view>

#include "nigiri/loader/assistance.h"
#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct loader_config {
  unsigned link_stop_distance_{100U};
  std::string default_tz_{};
  std::array<bool, kNumClasses> bikes_allowed_default_{};
  std::array<bool, kNumClasses> cars_allowed_default_{};
  bool extend_calendar_{false};
  std::string user_script_{};
  hash_set<std::string> base_paths_{};
};

struct loader_interface {
  virtual ~loader_interface();
  virtual bool applicable(dir const&) const = 0;
  virtual void load(loader_config const&,
                    source_idx_t,
                    dir const&,
                    timetable&,
                    hash_map<bitfield, bitfield_idx_t>&,
                    assistance_times*,
                    shapes_storage*) const = 0;
  virtual std::string_view name() const = 0;
};

}  // namespace nigiri::loader

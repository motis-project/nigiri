#pragma once

#include <string_view>

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct loader_config {
  unsigned link_stop_distance_{100U};
  std::string default_tz_;

  // finalize options
  bool adjust_footpaths_{true};
  bool merge_duplicates_{true};
  std::uint16_t max_footpath_length_{20};
};

struct loader_interface {
  virtual ~loader_interface();
  virtual bool applicable(dir const&) const = 0;
  virtual void load(loader_config const&,
                    source_idx_t,
                    dir const&,
                    timetable&) const = 0;
  virtual cista::hash_t hash(dir const&) const = 0;
  virtual std::string_view name() const = 0;
};

}  // namespace nigiri::loader

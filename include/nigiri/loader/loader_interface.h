#pragma once

#include <string_view>

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct loader_interface {
  virtual ~loader_interface();
  virtual bool applicable(dir const&) const = 0;
  virtual void load(source_idx_t, dir const&, timetable&) const = 0;
  virtual cista::hash_t hash(dir const&) const = 0;
  virtual std::string_view name() const = 0;
};

}  // namespace nigiri::loader

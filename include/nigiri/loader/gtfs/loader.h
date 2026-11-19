#pragma once

#include "nigiri/loader/loader_interface.h"

namespace nigiri::loader::gtfs {

struct gtfs_loader : public loader_interface {
  bool applicable(dir const&) const override;
  void load(loader_config const&,
            source_idx_t const,
            dir const&,
            timetable&,
            import_context&,
            assistance_times*,
            shapes_storage*) const override;
  std::string_view name() const override;
};

}  // namespace nigiri::loader::gtfs
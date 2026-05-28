#pragma once

#include "nigiri/loader/loader_interface.h"

namespace nigiri::loader::netex {

struct netex_loader : public loader_interface {
  bool applicable(dir const&) const override;
  void load(loader_config const&,
            source_idx_t const,
            dir const&,
            timetable&,
            hash_map<bitfield, bitfield_idx_t>&,
            assistance_times*,
            shapes_storage*) const override;
  std::string_view name() const override;
};

}  // namespace nigiri::loader::netex

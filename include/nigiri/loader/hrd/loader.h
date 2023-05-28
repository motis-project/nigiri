#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/loader_interface.h"

namespace nigiri::loader::hrd {

struct hrd_loader : public loader_interface {
  explicit hrd_loader(nigiri::loader::hrd::config c);
  bool applicable(dir const& d) const override;
  void load(loader_config const&,
            source_idx_t const src,
            dir const& d,
            timetable& tt) const override;
  cista::hash_t hash(dir const&) const override;
  nigiri::loader::hrd::config config_;
};

struct hrd_5_00_8_loader : public hrd_loader {
  hrd_5_00_8_loader();
  std::string_view name() const override;
};

struct hrd_5_20_26_loader : public hrd_loader {
  hrd_5_20_26_loader();
  std::string_view name() const override;
};

struct hrd_5_20_39_loader : public hrd_loader {
  hrd_5_20_39_loader();
  std::string_view name() const override;
};

struct hrd_5_20_avv_loader : public hrd_loader {
  hrd_5_20_avv_loader();
  std::string_view name() const override;
};

}  // namespace nigiri::loader::hrd
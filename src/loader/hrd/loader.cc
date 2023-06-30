#include "nigiri/loader/hrd/loader.h"

#include "nigiri/loader/hrd/load_timetable.h"

namespace nigiri::loader::hrd {

hrd_loader::hrd_loader(nigiri::loader::hrd::config c) : config_{std::move(c)} {}

bool hrd_loader::applicable(dir const& d) const {
  return nigiri::loader::hrd::applicable(config_, d);
}

void hrd_loader::load(loader_config const&,
                      source_idx_t const src,
                      dir const& d,
                      timetable& tt) const {
  return nigiri::loader::hrd::load_timetable(src, config_, d, tt);
}

cista::hash_t hrd_loader::hash(dir const& d) const {
  return nigiri::loader::hrd::hash(config_, d);
}

hrd_5_00_8_loader::hrd_5_00_8_loader() : hrd_loader{hrd_5_00_8} {}
std::string_view hrd_5_00_8_loader::name() const { return "hrd_5_00_8"; }

hrd_5_20_26_loader::hrd_5_20_26_loader() : hrd_loader{hrd_5_20_26} {}
std::string_view hrd_5_20_26_loader::name() const { return "hrd_5_20_26"; }

hrd_5_20_39_loader::hrd_5_20_39_loader() : hrd_loader{hrd_5_20_39} {}
std::string_view hrd_5_20_39_loader::name() const { return "hrd_5_20_39"; }

hrd_5_20_avv_loader::hrd_5_20_avv_loader() : hrd_loader{hrd_5_20_avv} {}
std::string_view hrd_5_20_avv_loader::name() const { return "hrd_5_20_avv"; }

}  // namespace nigiri::loader::hrd
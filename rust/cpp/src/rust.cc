#include "nigiri/rust.h"

#include "utl/to_vec.h"

#include "nigiri/loader/dir.h"

namespace fs = std::filesystem;
using namespace nigiri::loader;

std::string_view to_sv(rust::String const& s) { return {s.data(), s.size()}; }

std::unique_ptr<Timetable> new_timetable(rust::Vec<rust::String> const& paths) {
  auto loaders =
      std::vector<std::unique_ptr<nigiri::loader::loader_interface>>{};
  loaders.emplace_back(std::make_unique<nigiri::loader::gtfs::gtfs_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_00_8_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_26_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_39_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_avv_loader>());

  auto const dirs =
      utl::to_vec(paths, [](auto&& p) { return make_dir(fs::path{to_sv(p)}); });

  return std::make_unique<Timetable>();
}
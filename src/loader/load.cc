#include "nigiri/loader/load.h"

#include "fmt/std.h"

#include "utl/enumerate.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

std::vector<std::unique_ptr<loader_interface>> get_loaders() {
  auto loaders = std::vector<std::unique_ptr<loader_interface>>{};
  loaders.emplace_back(std::make_unique<gtfs::gtfs_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_00_8_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_26_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_39_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_avv_loader>());
  return loaders;
}

timetable load(
    std::vector<std::pair<std::filesystem::path, loader_config>> const& paths,
    finalize_options const& finalize_opt,
    interval<date::sys_days> const& date_range,
    assistance_times* a,
    shapes_storage_t* shapes,
    bool ignore) {
  auto const loaders = get_loaders();

  auto tt = timetable{};
  tt.date_range_ = date_range;
  register_special_stations(tt);

  auto bitfields = hash_map<bitfield, bitfield_idx_t>{};
  for (auto const [idx, in] : utl::enumerate(paths)) {
    auto const& [path, local_config] = in;
    auto const src = source_idx_t{idx};
    auto const dir = make_dir(path);
    auto const it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(*dir); });
    if (it != end(loaders)) {
      log(log_lvl::info, "loader.load", "loading {}", path);
      (*it)->load(local_config, src, *dir, tt, bitfields, a, shapes);
    } else if (!ignore) {
      throw utl::fail("no loader for {} found", path);
    } else {
      log(log_lvl::error, "loader.load", "no loader for {} found", path);
    }
  }

  finalize(tt, finalize_opt);

  return tt;
}

}  // namespace nigiri::loader
#include "nigiri/loader/load.h"

#include <tuple>

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

timetable load(std::vector<std::filesystem::path> const& paths,
               loader_config const& config,
               interval<date::sys_days> const& date_range) {
  auto const loaders = get_loaders();

  auto tt = timetable{};
  tt.date_range_ = date_range;
  register_special_stations(tt);

  for (auto const [idx, p] : utl::enumerate(paths)) {
    auto const src = source_idx_t{idx};
    auto const dir = make_dir(p);
    auto const loader_it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(*dir); });
    utl::verify(loader_it != end(loaders), "no loader for {} found",
                p.string());
    log(log_lvl::info, "loader.load", "loading {}", p.string());
    (*loader_it)->load(config, src, *dir, tt);
  }

  finalize(tt);

  return tt;
}

}  // namespace nigiri::loader
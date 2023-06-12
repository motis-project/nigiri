#include <vector>

#include "date/date.h"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;

int main(int ac, char** av) {
  if (ac != 2) {
    fmt::print("usage: {} [TIMETABLE_PATH]\n",
               ac == 0U ? "nigiri-server" : av[0]);
    return 1;
  }

  auto loaders = std::vector<std::unique_ptr<loader_interface>>{};
  loaders.emplace_back(std::make_unique<gtfs::gtfs_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_00_8_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_26_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_39_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_avv_loader>());

  auto const src = source_idx_t{0U};
  auto const tt_path = std::filesystem::path{av[1]};
  auto const d = make_dir(tt_path);

  auto const c =
      utl::find_if(loaders, [&](auto&& l) { return l->applicable(*d); });
  utl::verify(c != end(loaders), "no loader applicable to {}", tt_path);
  log(log_lvl::info, "main", "loading nigiri timetable with configuration {}",
      (*c)->name());

  timetable tt;
  tt.date_range_ = {date::sys_days{April / 1 / 2023},
                    date::sys_days{December / 1 / 2023}};
  register_special_stations(tt);
  (*c)->load({}, src, *d, tt);
  finalize(tt);
}
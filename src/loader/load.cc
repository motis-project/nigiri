#include "nigiri/loader/load.h"

#include "fmt/std.h"

#include "utl/enumerate.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/import_context.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/netex/loader.h"
#include "nigiri/timetable.h"
#include "nigiri/loader/gtfs/validate.h"
#include "nigiri/loader/gtfs/load_timetable.h"

namespace nigiri::loader {

std::vector<std::unique_ptr<loader_interface>> get_loaders() {
  auto loaders = std::vector<std::unique_ptr<loader_interface>>{};
  loaders.emplace_back(std::make_unique<gtfs::gtfs_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_00_8_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_26_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_39_loader>());
  loaders.emplace_back(std::make_unique<hrd::hrd_5_20_avv_loader>());
  loaders.emplace_back(std::make_unique<netex::netex_loader>());
  return loaders;
}

timetable load(std::vector<timetable_source> const& sources,
               finalize_options const& finalize_opt,
               interval<date::sys_days> const& date_range,
               assistance_times* a,
               shapes_storage* shapes,
               bool ignore) {
  auto const loaders = get_loaders();

  auto const accepted_sources = gtfs::validate_sources(sources);
  auto tt = timetable{};
  tt.date_range_ = date_range;
  tt.n_sources_ = static_cast<cista::base_t<source_idx_t>>(accepted_sources.size());
  auto ctx = import_context{tt};
  register_special_stations(tt, ctx);
  auto const progress_tracker = utl::get_active_progress_tracker();
  for (auto const [idx, in] : utl::enumerate(accepted_sources)) {
    auto const& [tag, path, local_config] = in;
    auto const is_in_memory = path.starts_with("\n#");
    auto const src = source_idx_t{idx};
    auto const dir = is_in_memory
                         // hack to load strings in integration tests
                         ? std::make_unique<mem_dir>(mem_dir::read(path))
                         : gtfs::find_gtfs_root(make_dir(path));
    auto const it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(*dir); });
    if (it != end(loaders)) {
      if (!is_in_memory) {
        log(log_lvl::info, "loader.load", "loading {}", path);
      }
      progress_tracker->context(std::string{tag});
      try {
        (*it)->load(local_config, src, *dir, tt, ctx, a, shapes);
      } catch (std::exception const& e) {
        throw utl::fail("failed to load {}: {}", path, e.what());
      }
      progress_tracker->context("");
    } else if (!ignore) {
      throw utl::fail("no loader for {} found", path);
    } else {
      log(log_lvl::error, "loader.load", "no loader for {} found", path);
    }
  }

  progress_tracker->status("Finalizing").out_bounds(98.F, 100.F).in_high(1);
  auto opt = finalize_opt;
  for (auto const& s : accepted_sources) {
    opt.src_tags_.push_back(s.tag_);
  }
  finalize(tt, ctx, opt);

  return tt;
}

timetable load(std::vector<dir_source> const& sources,
               interval<date::sys_days> const& date_range,
               finalize_options const& opt,
               assistance_times* a,
               shapes_storage* shapes) {
  auto const loaders = get_loaders();

  auto tt = timetable{};
  tt.date_range_ = date_range;
  tt.n_sources_ = static_cast<cista::base_t<source_idx_t>>(sources.size());
  auto ctx = import_context{tt};
  register_special_stations(tt, ctx);
  for (auto const [idx, s] : utl::enumerate(sources)) {
    auto const it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(s.dir_); });
    utl::verify(it != end(loaders), "no loader for source {} found", idx);
    (*it)->load(s.loader_config_, source_idx_t{idx}, s.dir_, tt, ctx, a,
                shapes);
  }
  finalize(tt, ctx, opt);

  return tt;
}

timetable load(dir const& d,
               interval<date::sys_days> const& date_range,
               finalize_options const& opt,
               assistance_times* a,
               shapes_storage* shapes) {
  return load(std::vector<dir_source>{{d}}, date_range, opt, a, shapes);
}

}  // namespace nigiri::loader

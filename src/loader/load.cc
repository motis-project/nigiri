#include <cassert>
#include <chrono>

#include "nigiri/loader/load.h"

#include "cista/hash.h"

#include "fmt/std.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/netex/loader.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace fs = std::filesystem;

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

using last_write_time_t = cista::strong<std::int64_t, struct _last_write_time>;
using source_path_t = cista::basic_string<char const*>;

struct change_detector {
  vector_map<source_idx_t, source_path_t> source_paths_;
  vector_map<source_idx_t, std::uint64_t> source_config_hashes_;
  vector_map<source_idx_t, last_write_time_t> last_write_times_;
};

timetable load(std::vector<timetable_source> const& sources,
               finalize_options const& finalize_opt,
               interval<date::sys_days> const& date_range,
               assistance_times* a,
               shapes_storage* shapes,
               bool ignore) {
  auto const loaders = get_loaders();
  auto const cache_path = fs::path{"cache"};
  auto const cache_metadata_path = cache_path / "meta.bin";

  fs::create_directories(cache_path);
  auto chg = change_detector{};
  for (auto const& in : sources) {
    auto const& [tag, path, local_config] = in;

    auto const last_write_time = fs::last_write_time(path);
    auto const timestamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::file_clock::to_sys(last_write_time).time_since_epoch())
            .count();
    chg.source_paths_.emplace_back(path);
    chg.source_config_hashes_.emplace_back(
        cista::hashing<loader_config>{}(local_config));
    chg.last_write_times_.emplace_back(last_write_time_t{timestamp});
  }

  try {
    cista::write(cache_metadata_path, chg);
  } catch (std::exception const& e) {
    log(log_lvl::error, "loader.load", "couldn't write cache metadata to {}",
        cache_metadata_path);
  }

  auto tt = timetable{};
  tt.date_range_ = date_range;
  tt.n_sources_ = static_cast<cista::base_t<source_idx_t>>(sources.size());
  register_special_stations(tt);
  auto const progress_tracker = utl::get_active_progress_tracker();
  for (auto const [idx, in] : utl::enumerate(sources)) {
    auto const local_cache_path = cache_path / fmt::format("tt{:d}", idx);
    auto const& [tag, path, local_config] = in;
    auto const is_in_memory = path.starts_with("\n#");
    auto const src = source_idx_t{idx};
    auto const dir = is_in_memory
                         // hack to load strings in integration tests
                         ? std::make_unique<mem_dir>(mem_dir::read(path))
                         : make_dir(path);
    auto const it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(*dir); });
    if (it != end(loaders)) {
      if (!is_in_memory) {
        log(log_lvl::info, "loader.load", "loading {}", path);
      }
      progress_tracker->context(std::string{tag});
      auto bitfields = hash_map<bitfield, bitfield_idx_t>{};
      for (auto const [idx_, bf] : utl::enumerate(tt.bitfields_)) {
        auto new_idx =
            utl::get_or_create(bitfields, bf, [&]() { return idx_; });
        assert(new_idx == idx_);  // bitfields must be unique in the timetable
      }
      try {
        (*it)->load(local_config, src, *dir, tt, bitfields, a, shapes);
      } catch (std::exception const& e) {
        throw utl::fail("failed to load {}: {}", path, e.what());
      }
      fs::create_directories(local_cache_path);
      if (shapes != nullptr) {
        auto shape_store =
            std::make_unique<shapes_storage>(local_cache_path, shapes->mode_);
        for (auto i = 0U; i < shapes->data_.size(); ++i) {
          auto const idx = shape_idx_t{i};
          shape_store->data_.emplace_back(shapes->data_[idx]);
        }
        for (auto const& e : shapes->offsets_) {
          shape_store->offsets_.emplace_back(e);
        }
        for (auto const& e : shapes->trip_offset_indices_) {
          shape_store->trip_offset_indices_.emplace_back(e);
        }
        for (auto const& e : shapes->route_bboxes_) {
          shape_store->route_bboxes_.emplace_back(e);
        }
        for (auto const& e : shapes->route_segment_bboxes_) {
          shape_store->route_segment_bboxes_.emplace_back(e);
        }
      }
      tt.write(local_cache_path / "tt.bin");
      progress_tracker->context("");
    } else if (!ignore) {
      throw utl::fail("no loader for {} found", path);
    } else {
      log(log_lvl::error, "loader.load", "no loader for {} found", path);
    }
  }

  progress_tracker->status("Finalizing").out_bounds(98.F, 100.F).in_high(1);
  finalize(tt, finalize_opt);

  return tt;
}

}  // namespace nigiri::loader

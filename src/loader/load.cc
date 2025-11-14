#include <cassert>
#include <chrono>

#include "nigiri/loader/load.h"

#include "cista/containers/string.h"
#include "cista/hash.h"
#include "cista/strong.h"

#include "fmt/std.h"

#include "utl/enumerate.h"
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

struct load_result {
  timetable tt_;
  std::unique_ptr<shapes_storage> shapes_;
};

struct loading_source_node {
  load_result load_from_source(dir* const dir, auto const it) const {
    // create local state
    auto const& [tag, path, local_config] = source_;
    auto bitfields = hash_map<bitfield, bitfield_idx_t>{};
    auto shape_store =
        has_shapes_ ? std::make_unique<shapes_storage>(
                          tmp_shapes_dir_, cista::mmap::protection::WRITE)
                    : nullptr;
    auto tt = timetable{};
    tt.date_range_ = date_range_;
    tt.n_sources_ = 1U;
    /* Load file */
    try {
      (*it)->load(local_config, source_idx_t{0}, *dir, tt, bitfields, a_,
                  shape_store.get());
    } catch (std::exception const& e) {
      throw utl::fail("failed to load {}: {}", path, e.what());
    }
    return load_result{.tt_ = tt, .shapes_ = std::move(shape_store)};
  }

  std::optional<load_result> load() const {
    auto const loaders = get_loaders();
    auto const& [tag, path, local_config] = source_;
    auto const is_in_memory = path.starts_with("\n#");
    auto const dir = is_in_memory
                         // hack to load strings in integration tests
                         ? std::make_unique<mem_dir>(mem_dir::read(path))
                         : make_dir(path);
    auto const it =
        utl::find_if(loaders, [&](auto&& l) { return l->applicable(*dir); });
    if (it != std::end(loaders)) {
      if (!is_in_memory) {
        log(log_lvl::info, "loader.load", "loading {}", path);
      }

      return load_from_source(dir.get(), it);

    } else if (!ignore_missing_loader_) {
      throw utl::fail("no loader for {} found", path);
    } else {
      log(log_lvl::error, "loader.load", "no loader for {} found", path);
    }
    return std::optional<load_result>{};
  }

  fs::path const tmp_shapes_dir_;
  timetable_source const source_;
  assistance_times* const a_;
  interval<date::sys_days> const date_range_;
  bool const has_shapes_;
  bool const ignore_missing_loader_;
};

std::optional<load_result> load_from_cache(
    fs::path const local_cache_path,
    interval<date::sys_days> const& date_range) {
  auto cached_timetable = timetable{};
  try {
    cached_timetable = *cista::read<timetable>(local_cache_path / "tt.bin");
  } catch (std::exception const& e) {
    log(log_lvl::info, "loader.load", "no cached timetable at {} found",
        local_cache_path / "tt.bin");
    return std::optional<load_result>{};
  }
  cached_timetable.resolve();
  if (cached_timetable.date_range_ != date_range) {
    return std::optional<load_result>{};
  }
  auto cached_shape_store = std::make_unique<shapes_storage>(
      local_cache_path, cista::mmap::protection::READ);
  return load_result{.tt_ = cached_timetable,
                     .shapes_ = std::move(cached_shape_store)};
}

using last_write_time_t = cista::strong<std::int64_t, struct _last_write_time>;
using source_path_t = cista::basic_string<char const*>;
using node_idx_t = cista::strong<std::uint32_t, struct _node_idx>;

struct loading_tree {
  node_idx_t n_nodes() const {
    return parent_.size() > 0 ? node_idx_t{2 * parent_.size() - 1}
                              : node_idx_t{0};
  }

  node_idx_t leaf_index(source_idx_t const& src) const {
    // Index 0 is the default timetable
    // After that, the source nodes follow
    // Then all intermediate nodes
    return node_idx_t{cista::to_idx(src) + 1};
  }

  node_idx_t insert_node(uint64_t cache_idx = 0) {
    return insert_node(node_idx_t::invalid(), node_idx_t::invalid(), cache_idx);
  }

  node_idx_t insert_node(node_idx_t left,
                         node_idx_t right,
                         uint64_t cache_idx = 0,
                         node_idx_t parent = node_idx_t::invalid()) {
    auto node_idx = node_idx_t{parent_.size()};

    parent_.emplace_back(parent);
    left_.emplace_back(left);
    right_.emplace_back(right);

    // Avoid accidental duplication
    cache_idx = cache_idx < cista::to_idx(n_nodes()) ? cista::to_idx(node_idx)
                                                     : cache_idx;
    cache_indices_.emplace_back(cache_idx);

    if (parent != node_idx_t::invalid()) {
      if (left_[parent] == node_idx_t::invalid()) {
        left_[parent] = node_idx;
      } else if (right_[parent] == node_idx_t::invalid()) {
        right_[parent] = node_idx;
      } else {
        throw utl::fail("invalid parent {} for {}", parent, node_idx);
      }
    }
    if (left != node_idx_t::invalid()) {
      if (parent_[left] == node_idx_t::invalid()) {
        parent_[left] = node_idx;
      } else {
        throw utl::fail("invalid left child {} for {}", left, node_idx);
      }
    }
    if (right != node_idx_t::invalid()) {
      if (parent_[right] == node_idx_t::invalid()) {
        parent_[right] = node_idx;
      } else {
        throw utl::fail("invalid right child {} for {}", right, node_idx);
      }
    }
    return node_idx;
  }

  vector_map<source_idx_t, source_path_t> source_paths_;
  vector_map<source_idx_t, std::uint64_t> source_config_hashes_;
  vector_map<source_idx_t, last_write_time_t> last_write_times_;

  vector_map<node_idx_t, node_idx_t> parent_;
  vector_map<node_idx_t, node_idx_t> left_;
  vector_map<node_idx_t, node_idx_t> right_;

  vector_map<node_idx_t, uint64_t> cache_indices_;
};

std::optional<load_result> run_loading(
    loading_tree const& l,
    node_idx_t const root,
    bitvec_map<node_idx_t> const& needs_recomputation,
    vector_map<source_idx_t, loading_source_node> const& source_loading_nodes,
    fs::path const& cache_path,
    interval<date::sys_days> const& date_range,
    auto& progress_tracker) {
  auto const local_cache_path =
      cache_path / fmt::format("tt{:d}", l.cache_indices_[root]);
  if (root == node_idx_t{0}) {
    auto tt = timetable{};
    tt.date_range_ = date_range;
    tt.n_sources_ = 0U;
    register_special_stations(tt);
    auto shape_store = std::make_unique<shapes_storage>(
        local_cache_path, cista::mmap::protection::WRITE);
    return load_result{.tt_ = tt, .shapes_ = std::move(shape_store)};
  }
  if (!needs_recomputation[root]) {
    auto result = load_from_cache(local_cache_path, date_range);
    if (result.has_value()) {
      return result;
    }
  }
  fs::create_directories(local_cache_path);
  auto const is_merge_node = cista::to_idx(root) > l.source_paths_.size();
  if (is_merge_node) {
    auto const left = l.left_[root];
    auto const right = l.right_[root];
    auto left_result =
        run_loading(l, left, needs_recomputation, source_loading_nodes,
                    cache_path, date_range, progress_tracker);
    auto right_result =
        run_loading(l, right, needs_recomputation, source_loading_nodes,
                    cache_path, date_range, progress_tracker);
    if (!left_result.has_value()) {
      return right_result;
    } else if (!right_result.has_value()) {
      return left_result;
    }
    auto& left_tt = left_result.value().tt_;
    auto left_shapes = std::move(left_result.value().shapes_);
    auto const& right_tt = right_result.value().tt_;
    auto right_shapes = std::move(right_result.value().shapes_);

    progress_tracker->status(
        fmt::format("Merging timetables '{:d}' and '{:d}'...",
                    l.cache_indices_[left], l.cache_indices_[right]));
    left_tt.merge(right_tt);

    /* Save snapshot */
    progress_tracker->status(
        fmt::format("Saving cache for '{:d}'...", l.cache_indices_[root]));
    left_tt.write(local_cache_path / "tt.bin");
    assert((left_shapes == nullptr) == (right_shapes == nullptr));
    if (left_shapes == nullptr || right_shapes == nullptr) {
      return load_result{.tt_ = left_tt, .shapes_ = nullptr};
    }

    auto shape_store = std::make_unique<shapes_storage>(
        local_cache_path, cista::mmap::protection::WRITE);
    shape_store->add(left_shapes.get());
    shape_store->add(right_shapes.get());

    return load_result{.tt_ = left_tt, .shapes_ = std::move(shape_store)};
  }
  auto const src = source_idx_t{cista::to_idx(root) - 1};
  auto const node = source_loading_nodes[src];
  progress_tracker->context(std::string{node.source_.tag_});
  progress_tracker->status("Loading timetable data...");

  auto result = node.load();
  if (result.has_value()) {
    // node.load() writes the shapes onto the filesystem
    result.value().tt_.write(local_cache_path / "tt.bin");
  }
  progress_tracker->context("");
  return result;
}

timetable load(std::vector<timetable_source> const& sources,
               finalize_options const& finalize_opt,
               interval<date::sys_days> const& date_range,
               assistance_times* a,
               shapes_storage* shapes,
               bool ignore) {
  auto const cache_path = fs::path{"cache"};
  auto const cache_metadata_path = cache_path / "meta.bin";

  fs::create_directories(cache_path);
  auto chg = loading_tree{};
  chg.insert_node();  // default timetable node
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
    chg.insert_node();
  }
  auto saved_changes = loading_tree{};
  try {
    saved_changes = *cista::read<loading_tree>(cache_metadata_path);
  } catch (std::exception const& e) {
    log(log_lvl::info, "loader.load", "no cache metadata at {} found",
        cache_metadata_path);
  }
  auto needs_recomputation = bitvec_map<node_idx_t>{};
  needs_recomputation.resize(cista::to_idx(chg.n_nodes()));
  for (auto const [idx, path] : utl::enumerate(chg.source_paths_)) {
    auto const src = source_idx_t{idx};
    needs_recomputation.set(
        chg.leaf_index(src),
        chg.source_paths_.size() != saved_changes.source_paths_.size() ||
            path != saved_changes.source_paths_[src] ||
            chg.last_write_times_[src] !=
                saved_changes.last_write_times_[src] ||
            chg.source_config_hashes_[src] !=
                saved_changes.source_config_hashes_[src]);
  }

  auto root = node_idx_t{0};
  for (auto const [idx, in] : utl::enumerate(chg.source_paths_)) {
    auto const src = source_idx_t{idx};
    auto const left = root;
    auto const right = chg.leaf_index(src);
    root = chg.insert_node(left, right);
    needs_recomputation.set(
        root, needs_recomputation[left] || needs_recomputation[right]);
  }

  try {
    cista::write(cache_metadata_path, chg);
  } catch (std::exception const& e) {
    log(log_lvl::error, "loader.load", "couldn't write cache metadata to {}",
        cache_metadata_path);
  }

  auto source_loading_nodes = vector_map<source_idx_t, loading_source_node>{};
  for (auto const [idx, in] : utl::enumerate(sources)) {
    auto const local_cache_path =
        cache_path /
        fmt::format("tt{:d}",
                    chg.cache_indices_[chg.leaf_index(source_idx_t{idx})]);
    auto const& node = loading_source_node{.tmp_shapes_dir_ = local_cache_path,
                                           .source_ = in,
                                           .a_ = a,
                                           .date_range_ = date_range,
                                           .has_shapes_ = shapes != nullptr,
                                           .ignore_missing_loader_ = ignore};
    source_loading_nodes.push_back(node);
  }
  auto const progress_tracker = utl::get_active_progress_tracker();
  auto const& result =
      run_loading(chg, root, needs_recomputation, source_loading_nodes,
                  cache_path, date_range, progress_tracker);
  assert(result.has_value());  // Guaranteed since the default empty timetable
                               // cannot fail to load
  auto tt = result.value().tt_;
  shapes->add(result.value().shapes_.get());

  progress_tracker->status("Finalizing").out_bounds(98.F, 100.F).in_high(1);
  finalize(tt, finalize_opt);

  return tt;
}

}  // namespace nigiri::loader

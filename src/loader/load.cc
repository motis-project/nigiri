#include <cassert>
#include <chrono>
#include <ranges>

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
namespace vw = std::views;

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

std::pair<timetable, std::unique_ptr<shapes_storage>> load_from_source(
    uint64_t const idx,
    dir* const dir,
    assistance_times* a,
    auto const it,
    std::vector<timetable_source> const& sources,
    interval<date::sys_days> const& date_range,
    fs::path const cache_path,
    shapes_storage const* shapes) {
  // create local state
  auto const& [tag, path, local_config] = sources[idx];
  auto const load_local_cache_path =
      cache_path / fmt::format("tt{:d}", idx + sources.size());
  auto bitfields = hash_map<bitfield, bitfield_idx_t>{};
  auto shape_store = shapes != nullptr
                         ? std::make_unique<shapes_storage>(
                               load_local_cache_path, shapes->mode_)
                         : nullptr;
  auto tt = timetable{};
  tt.date_range_ = date_range;
  tt.n_sources_ = 1U;
  /* Load file */
  try {
    (*it)->load(local_config, source_idx_t{0}, *dir, tt, bitfields, a,
                shape_store.get());
  } catch (std::exception const& e) {
    throw utl::fail("failed to load {}: {}", path, e.what());
  }
  tt.write(load_local_cache_path / "tt.bin");
  return std::make_pair(tt, std::move(shape_store));
}

using last_write_time_t = cista::strong<std::int64_t, struct _last_write_time>;
using source_path_t = cista::basic_string<char const*>;

struct change_detector {
  vector_map<source_idx_t, source_path_t> source_paths_;
  vector_map<source_idx_t, std::uint64_t> source_config_hashes_;
  vector_map<source_idx_t, last_write_time_t> last_write_times_;
};

struct index_mapping {
  location_idx_t const location_idx_offset_;
  trip_direction_string_idx_t const trip_direction_string_idx_offset_;

  index_mapping(timetable const& first_tt)
      : location_idx_offset_{first_tt.n_locations()},
        trip_direction_string_idx_offset_{
            first_tt.trip_direction_strings_.size()} {}

  auto map(location_idx_t const& i) const {
    return i != location_idx_t::invalid() ? i + location_idx_offset_
                                          : location_idx_t::invalid();
  }
  auto map(trip_direction_string_idx_t const& i) const {
    return i != trip_direction_string_idx_t::invalid()
               ? i + trip_direction_string_idx_offset_
               : trip_direction_string_idx_t::invalid();
  }
  auto map(trip_direction_t const& i) const {
    return i.apply([&](auto const& d) -> trip_direction_t {
      return trip_direction_t{map(d)};
    });
  }

  template <typename T>
  auto map(T const& i) const {
    return i;
  }
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
  auto saved_changes = change_detector{};
  try {
    saved_changes = *cista::read<change_detector>(cache_metadata_path);
  } catch (std::exception const& e) {
    log(log_lvl::info, "loader.load", "no cache metadata at {} found",
        cache_metadata_path);
  }
  auto first_recomputed_source = source_idx_t{chg.source_paths_.size()};
  for (auto const [idx, path] : utl::enumerate(chg.source_paths_)) {
    auto const src = source_idx_t{idx};
    if (chg.source_paths_.size() != saved_changes.source_paths_.size()) {
      first_recomputed_source = src;
      break;
    }
    if (path != saved_changes.source_paths_[src] ||
        chg.last_write_times_[src] != saved_changes.last_write_times_[src]) {
      first_recomputed_source = src;
      break;
    }
    if (chg.source_config_hashes_[src] !=
        saved_changes.source_config_hashes_[src]) {
      first_recomputed_source = src;
      break;
    }
  }

  auto tt = timetable{};
  auto const progress_tracker = utl::get_active_progress_tracker();
  for (auto i = first_recomputed_source; i >= 0; --i) {
    if (i == 0) {
      tt.date_range_ = date_range;
      tt.n_sources_ = 0U;
      register_special_stations(tt);
      break;
    }
    auto const prev = i - 1;
    auto const local_cache_path =
        cache_path / fmt::format("tt{:d}", to_idx(prev));
    auto cached_timetable = timetable{};
    try {
      cached_timetable = *cista::read<timetable>(local_cache_path / "tt.bin");
    } catch (std::exception const& e) {
      log(log_lvl::info, "loader.load", "no cached timetable at {} found",
          local_cache_path / "tt.bin");
      continue;
    }
    cached_timetable.resolve();
    if (cached_timetable.date_range_ != date_range) {
      continue;
    }
    first_recomputed_source = i;
    tt = cached_timetable;
    auto cached_shape_store = std::make_unique<shapes_storage>(
        local_cache_path, cista::mmap::protection::READ);
    shapes->add(cached_shape_store.get());
    break;
  }

  try {
    cista::write(cache_metadata_path, chg);
  } catch (std::exception const& e) {
    log(log_lvl::error, "loader.load", "couldn't write cache metadata to {}",
        cache_metadata_path);
  }

  for (auto const [idx, in] : utl::enumerate(sources)) {
    auto const local_cache_path = cache_path / fmt::format("tt{:d}", idx);
    auto const src = source_idx_t{idx};
    if (src < first_recomputed_source) {
      continue;
    }
    auto const& [tag, path, local_config] = in;
    auto const is_in_memory = path.starts_with("\n#");
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
      progress_tracker->status("Merging...");
      auto bitfields_ = hash_map<bitfield, bitfield_idx_t>{};
      for (auto const [idx_, bf] : utl::enumerate(tt.bitfields_)) {
        auto new_idx =
            utl::get_or_create(bitfields_, bf, [&]() { return idx_; });
        assert(new_idx == idx_);  // bitfields must be unique in the timetable
      }
      /* Save data to restore later */
      auto const old_bitfields = tt.bitfields_;
      auto const old_source_end_date = tt.src_end_date_;
      tt.src_end_date_ = old_source_end_date;
      auto const old_trip_id_to_idx = tt.trip_id_to_idx_;
      tt.trip_id_to_idx_ = old_trip_id_to_idx;
      auto const old_trip_ids = tt.trip_ids_;
      tt.trip_ids_ = old_trip_ids;
      auto const old_trip_id_strings = tt.trip_id_strings_;
      tt.trip_id_strings_ = old_trip_id_strings;
      auto const old_trip_id_src = tt.trip_id_src_;
      tt.trip_id_src_ = old_trip_id_src;
      auto const old_trip_direction_id = tt.trip_direction_id_;
      tt.trip_direction_id_ = old_trip_direction_id;
      auto const old_trip_route_id = tt.trip_route_id_;
      tt.trip_route_id_ = old_trip_route_id;
      auto const old_route_ids = tt.route_ids_;
      tt.route_ids_ = old_route_ids;
      auto const old_trip_transport_ranges = tt.trip_transport_ranges_;
      tt.trip_transport_ranges_ = old_trip_transport_ranges;
      auto const old_trip_stop_seq_numbers = tt.trip_stop_seq_numbers_;
      tt.trip_stop_seq_numbers_ = old_trip_stop_seq_numbers;
      auto const old_source_file_names = tt.source_file_names_;
      tt.source_file_names_ = old_source_file_names;
      auto const old_trip_debug = tt.trip_debug_;
      tt.trip_debug_ = old_trip_debug;
      auto const old_trip_short_names = tt.trip_short_names_;
      tt.trip_short_names_ = old_trip_short_names;
      auto const old_trip_display_names = tt.trip_display_names_;
      tt.trip_display_names_ = old_trip_display_names;
      auto const old_route_transport_ranges = tt.route_transport_ranges_;
      tt.route_transport_ranges_ = old_route_transport_ranges;
      auto const old_route_location_seq = tt.route_location_seq_;
      tt.route_location_seq_ = old_route_location_seq;
      auto const old_route_clasz = tt.route_clasz_;
      tt.route_clasz_ = old_route_clasz;
      auto const old_route_section_clasz = tt.route_section_clasz_;
      tt.route_section_clasz_ = old_route_section_clasz;
      auto const old_route_bikes_allowed = tt.route_bikes_allowed_;
      tt.route_bikes_allowed_ = old_route_bikes_allowed;
      auto const old_route_cars_allowed = tt.route_cars_allowed_;
      tt.route_cars_allowed_ = old_route_cars_allowed;
      auto const old_route_bikes_allowed_per_section =
          tt.route_bikes_allowed_per_section_;
      tt.route_bikes_allowed_per_section_ = old_route_bikes_allowed_per_section;
      auto const old_route_cars_allowed_per_section =
          tt.route_cars_allowed_per_section_;
      tt.route_cars_allowed_per_section_ = old_route_cars_allowed_per_section;
      auto const old_route_stop_time_ranges = tt.route_stop_time_ranges_;
      tt.route_stop_time_ranges_ = old_route_stop_time_ranges;
      auto const old_route_stop_times = tt.route_stop_times_;
      tt.route_stop_times_ = old_route_stop_times;
      auto const old_transport_first_dep_offset =
          tt.transport_first_dep_offset_;
      tt.transport_first_dep_offset_ = old_transport_first_dep_offset;
      auto const old_transport_traffic_days = tt.transport_traffic_days_;
      tt.transport_traffic_days_ = old_transport_traffic_days;
      auto const old_transport_route = tt.transport_route_;
      tt.transport_route_ = old_transport_route;
      auto const old_transport_to_trip_section = tt.transport_to_trip_section_;
      tt.transport_to_trip_section_ = old_transport_to_trip_section;
      auto const old_languages = tt.languages_;
      tt.languages_ = old_languages;
      auto const old_locations = tt.locations_;
      tt.locations_ = old_locations;
      auto const old_merged_trips = tt.merged_trips_;
      tt.merged_trips_ = old_merged_trips;
      auto const old_attributes = tt.attributes_;
      tt.attributes_ = old_attributes;
      auto const old_attribute_combinations = tt.attribute_combinations_;
      tt.attribute_combinations_ = old_attribute_combinations;
      auto const old_trip_direction_strings = tt.trip_direction_strings_;
      tt.trip_direction_strings_ = old_trip_direction_strings;
      auto const old_trip_directions = tt.trip_directions_;
      tt.trip_directions_ = old_trip_directions;
      auto const old_trip_lines = tt.trip_lines_;
      tt.trip_lines_ = old_trip_lines;
      auto const old_transport_section_attributes =
          tt.transport_section_attributes_;
      tt.transport_section_attributes_ = old_transport_section_attributes;
      auto const old_transport_section_providers =
          tt.transport_section_providers_;
      tt.transport_section_providers_ = old_transport_section_providers;
      auto const old_transport_section_directions =
          tt.transport_section_directions_;
      tt.transport_section_directions_ = old_transport_section_directions;
      auto const old_transport_section_lines = tt.transport_section_lines_;
      tt.transport_section_lines_ = old_transport_section_lines;
      auto const old_transport_section_route_colors =
          tt.transport_section_route_colors_;
      tt.transport_section_route_colors_ = old_transport_section_route_colors;
      auto const old_location_routes = tt.location_routes_;
      tt.location_routes_ = old_location_routes;
      auto const old_providers = tt.providers_;
      tt.providers_ = old_providers;
      auto const old_provider_id_to_idx = tt.provider_id_to_idx_;
      tt.provider_id_to_idx_ = old_provider_id_to_idx;
      auto const old_fares = tt.fares_;
      tt.fares_ = old_fares;
      auto const old_areas = tt.areas_;
      tt.areas_ = old_areas;
      auto const old_location_areas = tt.location_areas_;
      tt.location_areas_ = old_location_areas;
      auto const old_location_location_groups_size =
          tt.location_location_groups_.size();
      auto const old_location_location_groups = tt.location_location_groups_;
      tt.location_location_groups_ = old_location_location_groups;
      assert(old_location_location_groups.size() ==
                 old_location_location_groups_size &&
             tt.location_location_groups_.size() ==
                 old_location_location_groups_size);
      auto const old_location_group_locations_size =
          tt.location_group_locations_.size();
      auto const old_location_group_locations = tt.location_group_locations_;
      tt.location_group_locations_ = old_location_group_locations;
      assert(old_location_group_locations.size() ==
                 old_location_group_locations_size &&
             tt.location_group_locations_.size() ==
                 old_location_group_locations_size);
      auto const old_location_group_name = tt.location_group_name_;
      tt.location_group_name_ = old_location_group_name;
      auto const old_location_group_id = tt.location_group_id_;
      tt.location_group_id_ = old_location_group_id;
      auto const old_flex_area_bbox = tt.flex_area_bbox_;
      tt.flex_area_bbox_ = old_flex_area_bbox;
      auto const old_flex_area_id = tt.flex_area_id_;
      tt.flex_area_id_ = old_flex_area_id;
      auto const old_flex_area_src = tt.flex_area_src_;
      tt.flex_area_src_ = old_flex_area_src;

      auto const old_flex_area_outers = tt.flex_area_outers_;
      tt.flex_area_outers_ = old_flex_area_outers;
      auto const old_flex_area_inners = tt.flex_area_inners_;
      tt.flex_area_inners_ = old_flex_area_inners;
      auto const old_flex_area_name = tt.flex_area_name_;
      tt.flex_area_name_ = old_flex_area_name;
      auto const old_flex_area_desc = tt.flex_area_desc_;
      tt.flex_area_desc_ = old_flex_area_desc;
      auto const old_flex_area_rtree = tt.flex_area_rtree_;
      tt.flex_area_rtree_ = old_flex_area_rtree;
      auto const old_location_group_transports = tt.location_group_transports_;
      tt.location_group_transports_ = old_location_group_transports;
      auto const old_flex_area_transports = tt.flex_area_transports_;
      tt.flex_area_transports_ = old_flex_area_transports;
      auto const old_flex_transport_traffic_days =
          tt.flex_transport_traffic_days_;
      tt.flex_transport_traffic_days_ = old_flex_transport_traffic_days;
      auto const old_flex_transport_trip = tt.flex_transport_trip_;
      tt.flex_transport_trip_ = old_flex_transport_trip;
      auto const old_flex_transport_stop_time_windows =
          tt.flex_transport_stop_time_windows_;
      tt.flex_transport_stop_time_windows_ =
          old_flex_transport_stop_time_windows;
      auto const old_flex_transport_stop_seq = tt.flex_transport_stop_seq_;
      tt.flex_transport_stop_seq_ = old_flex_transport_stop_seq;
      auto const old_flex_stop_seq = tt.flex_stop_seq_;
      tt.flex_stop_seq_ = old_flex_stop_seq;
      auto const old_flex_transport_pickup_booking_rule =
          tt.flex_transport_pickup_booking_rule_;
      tt.flex_transport_pickup_booking_rule_ =
          old_flex_transport_pickup_booking_rule;
      auto const old_flex_transport_drop_off_booking_rule =
          tt.flex_transport_drop_off_booking_rule_;
      tt.flex_transport_drop_off_booking_rule_ =
          old_flex_transport_drop_off_booking_rule;
      auto const old_booking_rules = tt.booking_rules_;
      tt.booking_rules_ = old_booking_rules;
      auto const old_strings = tt.strings_;
      tt.strings_ = old_strings;
      auto const old_n_sources = tt.n_sources_;
      tt.n_sources_ = old_n_sources;
      /* Prepare timetable by emptying corrected fields */
      // Fields not used during loading
      assert(tt.locations_.footpaths_out_.size() == kNProfiles);
      for (auto const& i : tt.locations_.footpaths_out_) {
        assert(i.size() == 0);
      }
      assert(tt.locations_.footpaths_in_.size() == kNProfiles);
      for (auto const& i : tt.locations_.footpaths_in_) {
        assert(i.size() == 0);
      }
      assert(tt.fwd_search_lb_graph_.size() == kNProfiles);
      for (auto const& i : tt.fwd_search_lb_graph_) {
        assert(i.size() == 0);
      }
      assert(tt.bwd_search_lb_graph_.size() == kNProfiles);
      for (auto const& i : tt.bwd_search_lb_graph_) {
        assert(i.size() == 0);
      }
      assert(tt.flex_area_locations_.size() == 0);
      assert(tt.trip_train_nr_.size() == 0);
      assert(tt.initial_day_offset_.size() == 0);
      assert(tt.profiles_.size() == 0);
      assert(tt.date_range_ == date_range);
      auto result = load_from_source(idx, dir.get(), a, it, sources, date_range,
                                     cache_path, shapes);
      tt = result.first;
      auto shape_store = std::move(result.second);
      /* Save new data */
      auto const new_bitfields = tt.bitfields_;
      auto const new_source_end_date = tt.src_end_date_;
      auto const new_trip_id_to_idx = tt.trip_id_to_idx_;
      auto const new_trip_ids = tt.trip_ids_;
      auto const new_trip_id_strings = tt.trip_id_strings_;
      auto const new_trip_id_src = tt.trip_id_src_;
      auto const new_trip_direction_id = tt.trip_direction_id_;
      auto const new_trip_route_id = tt.trip_route_id_;
      auto const new_route_ids = tt.route_ids_;
      auto const new_trip_transport_ranges = tt.trip_transport_ranges_;
      auto const new_trip_stop_seq_numbers = tt.trip_stop_seq_numbers_;
      auto const new_source_file_names = tt.source_file_names_;
      auto const new_trip_debug = tt.trip_debug_;
      auto const new_trip_short_names = tt.trip_short_names_;
      auto const new_trip_display_names = tt.trip_display_names_;
      auto const new_route_transport_ranges = tt.route_transport_ranges_;
      auto const new_route_location_seq = tt.route_location_seq_;
      auto const new_route_clasz = tt.route_clasz_;
      auto const new_route_section_clasz = tt.route_section_clasz_;
      auto const new_route_bikes_allowed = tt.route_bikes_allowed_;
      auto const new_route_cars_allowed = tt.route_cars_allowed_;
      auto const new_route_bikes_allowed_per_section =
          tt.route_bikes_allowed_per_section_;
      auto const new_route_cars_allowed_per_section =
          tt.route_cars_allowed_per_section_;
      auto const new_route_stop_time_ranges = tt.route_stop_time_ranges_;
      auto const new_route_stop_times = tt.route_stop_times_;
      auto const new_transport_first_dep_offset =
          tt.transport_first_dep_offset_;
      auto const new_transport_traffic_days = tt.transport_traffic_days_;
      auto const new_transport_route = tt.transport_route_;
      auto const new_transport_to_trip_section = tt.transport_to_trip_section_;
      auto const new_languages = tt.languages_;
      auto const new_locations = tt.locations_;
      auto const new_merged_trips = tt.merged_trips_;
      auto const new_attributes = tt.attributes_;
      auto const new_attribute_combinations = tt.attribute_combinations_;
      auto const new_trip_direction_strings = tt.trip_direction_strings_;
      auto const new_trip_directions = tt.trip_directions_;
      auto const new_trip_lines = tt.trip_lines_;
      auto const new_transport_section_attributes =
          tt.transport_section_attributes_;
      auto const new_transport_section_providers =
          tt.transport_section_providers_;
      auto const new_transport_section_directions =
          tt.transport_section_directions_;
      auto const new_transport_section_lines = tt.transport_section_lines_;
      auto const new_transport_section_route_colors =
          tt.transport_section_route_colors_;
      auto const new_location_routes = tt.location_routes_;
      auto const new_providers = tt.providers_;
      auto const new_provider_id_to_idx = tt.provider_id_to_idx_;
      auto const new_fares = tt.fares_;
      auto const new_areas = tt.areas_;
      auto const new_location_areas = tt.location_areas_;
      auto const new_location_location_groups = tt.location_location_groups_;
      auto const new_location_group_locations = tt.location_group_locations_;
      auto const new_location_group_name = tt.location_group_name_;
      auto const new_location_group_id = tt.location_group_id_;
      auto const new_flex_area_bbox = tt.flex_area_bbox_;
      auto const new_flex_area_id = tt.flex_area_id_;
      auto const new_flex_area_src = tt.flex_area_src_;
      auto const new_flex_area_outers = tt.flex_area_outers_;
      auto const new_flex_area_inners = tt.flex_area_inners_;
      auto const new_flex_area_name = tt.flex_area_name_;
      auto const new_flex_area_desc = tt.flex_area_desc_;
      auto const new_flex_area_rtree = tt.flex_area_rtree_;
      auto const new_location_group_transports = tt.location_group_transports_;
      auto const new_flex_area_transports = tt.flex_area_transports_;
      auto const new_flex_transport_traffic_days =
          tt.flex_transport_traffic_days_;
      auto const new_flex_transport_trip = tt.flex_transport_trip_;
      auto const new_flex_transport_stop_time_windows =
          tt.flex_transport_stop_time_windows_;
      auto const new_flex_transport_stop_seq = tt.flex_transport_stop_seq_;
      auto const new_flex_stop_seq = tt.flex_stop_seq_;
      auto const new_flex_transport_pickup_booking_rule =
          tt.flex_transport_pickup_booking_rule_;
      auto const new_flex_transport_drop_off_booking_rule =
          tt.flex_transport_drop_off_booking_rule_;
      auto const new_booking_rules = tt.booking_rules_;
      auto const new_strings = tt.strings_;
      auto const new_n_sources = tt.n_sources_;
      progress_tracker->status("Saved new data");
      /* Restore old timetable */
      tt.bitfields_ = old_bitfields;
      tt.src_end_date_ = old_source_end_date;
      tt.trip_id_to_idx_ = old_trip_id_to_idx;
      tt.trip_ids_ = old_trip_ids;
      tt.trip_id_strings_ = old_trip_id_strings;
      tt.trip_id_src_ = old_trip_id_src;
      tt.trip_direction_id_ = old_trip_direction_id;
      tt.trip_route_id_ = old_trip_route_id;
      tt.route_ids_ = old_route_ids;
      tt.trip_transport_ranges_ = old_trip_transport_ranges;
      tt.trip_stop_seq_numbers_ = old_trip_stop_seq_numbers;
      tt.source_file_names_ = old_source_file_names;
      tt.trip_debug_ = old_trip_debug;
      tt.trip_short_names_ = old_trip_short_names;
      tt.trip_display_names_ = old_trip_display_names;
      tt.route_transport_ranges_ = old_route_transport_ranges;
      tt.route_location_seq_ = old_route_location_seq;
      tt.route_clasz_ = old_route_clasz;
      tt.route_section_clasz_ = old_route_section_clasz;
      tt.route_bikes_allowed_ = old_route_bikes_allowed;
      tt.route_cars_allowed_ = old_route_cars_allowed;
      tt.route_bikes_allowed_per_section_ = old_route_bikes_allowed_per_section;
      tt.route_cars_allowed_per_section_ = old_route_cars_allowed_per_section;
      tt.route_stop_time_ranges_ = old_route_stop_time_ranges;
      tt.route_stop_times_ = old_route_stop_times;
      tt.transport_first_dep_offset_ = old_transport_first_dep_offset;
      tt.transport_traffic_days_ = old_transport_traffic_days;
      tt.transport_route_ = old_transport_route;
      tt.transport_to_trip_section_ = old_transport_to_trip_section;
      tt.merged_trips_ = old_merged_trips;
      tt.attributes_ = old_attributes;
      tt.attribute_combinations_ = old_attribute_combinations;
      tt.trip_direction_strings_ = old_trip_direction_strings;
      tt.trip_directions_ = old_trip_directions;
      tt.trip_lines_ = old_trip_lines;
      tt.transport_section_attributes_ = old_transport_section_attributes;
      tt.transport_section_providers_ = old_transport_section_providers;
      tt.transport_section_directions_ = old_transport_section_directions;
      tt.transport_section_lines_ = old_transport_section_lines;
      tt.transport_section_route_colors_ = old_transport_section_route_colors;
      tt.languages_ = old_languages;
      tt.locations_ = old_locations;
      tt.location_routes_ = old_location_routes;
      tt.providers_ = old_providers;
      tt.provider_id_to_idx_ = old_provider_id_to_idx;
      tt.fares_ = old_fares;
      tt.areas_ = old_areas;
      tt.location_areas_ = old_location_areas;
      tt.location_location_groups_ = old_location_location_groups;
      tt.location_group_locations_ = old_location_group_locations;

      assert(old_location_location_groups.size() ==
                 old_location_location_groups_size &&
             tt.location_location_groups_.size() ==
                 old_location_location_groups_size);
      assert(old_location_group_locations.size() ==
                 old_location_group_locations_size &&
             tt.location_group_locations_.size() ==
                 old_location_group_locations_size);

      tt.location_group_name_ = old_location_group_name;
      tt.location_group_id_ = old_location_group_id;
      tt.flex_area_bbox_ = old_flex_area_bbox;
      tt.flex_area_id_ = old_flex_area_id;
      tt.flex_area_src_ = old_flex_area_src;
      tt.flex_area_outers_ = old_flex_area_outers;
      tt.flex_area_inners_ = old_flex_area_inners;
      tt.flex_area_name_ = old_flex_area_name;
      tt.flex_area_desc_ = old_flex_area_desc;
      tt.flex_area_rtree_ = old_flex_area_rtree;
      tt.location_group_transports_ = old_location_group_transports;
      tt.flex_area_transports_ = old_flex_area_transports;
      tt.flex_transport_traffic_days_ = old_flex_transport_traffic_days;
      tt.flex_transport_trip_ = old_flex_transport_trip;
      tt.flex_transport_stop_time_windows_ =
          old_flex_transport_stop_time_windows;
      tt.flex_transport_stop_seq_ = old_flex_transport_stop_seq;
      tt.flex_stop_seq_ = old_flex_stop_seq;
      tt.flex_transport_pickup_booking_rule_ =
          old_flex_transport_pickup_booking_rule;
      tt.flex_transport_drop_off_booking_rule_ =
          old_flex_transport_drop_off_booking_rule;
      tt.booking_rules_ = old_booking_rules;
      tt.strings_ = old_strings;
      tt.n_sources_ = old_n_sources;
      /* Add new data and adjust references */
      auto const im = index_mapping(tt);
      /*	bitfields	*/
      auto corrected_indices = vector_map<bitfield_idx_t, bitfield_idx_t>{};
      for (auto const& [idx_, bf] : utl::enumerate(new_bitfields)) {
        auto adjusted_idx = utl::get_or_create(
            bitfields_, bf, [&]() { return tt.register_bitfield(bf); });
        corrected_indices.emplace_back(adjusted_idx);
      }
      /*       string_idx_t	*/
      auto string_map = vector_map<string_idx_t, string_idx_t>{};
      for (auto const& [idx_, s] : utl::enumerate(new_strings.strings_)) {
        auto new_idx = tt.strings_.store(s.view());
        string_map.push_back(new_idx);
      }
      /*	 sources	*/
      auto const source_idx_offset = source_idx_t{tt.src_end_date_.size()};
      for (auto const& i : new_source_end_date) {
        tt.src_end_date_.push_back(i);
      }
      auto const source_file_names_offset =
          source_file_idx_t{tt.source_file_names_.size()};
      for (auto const& i : new_source_file_names) {
        tt.source_file_names_.emplace_back(i);
      }
      for (auto const& i : new_trip_debug) {
        auto entry = tt.trip_debug_.emplace_back();
        for (auto const& j : i) {
          auto debug =
              trip_debug{j.source_file_idx_ != source_file_idx_t::invalid()
                             ? j.source_file_idx_ + source_file_names_offset
                             : source_file_idx_t::invalid(),
                         j.line_number_from_, j.line_number_to_};
          entry.emplace_back(debug);
        }
      }
      /*	 languages	*/
      auto const language_offset = language_idx_t{tt.languages_.size()};
      for (auto const& i : new_languages) {
        tt.languages_.emplace_back(i);
      }
      /*       location_idx_t	*/
      auto const locations_offset = location_idx_t{tt.n_locations()};
      auto const location_group_offset =
          location_group_idx_t{tt.location_group_name_.size()};
      auto const alt_name_idx_offset =
          alt_name_idx_t{tt.locations_.alt_name_strings_.size()};
      auto const timezones_offset =
          timezone_idx_t{tt.locations_.timezones_.size()};
      auto const trip_offset = trip_idx_t{tt.trip_ids_.size()};
      auto const route_idx_offset = route_idx_t{tt.n_routes()};
      {  // merge locations struct
        auto&& loc = tt.locations_;
        for (auto const& i : new_locations.location_id_to_idx_) {
          auto loc_id = i.first;
          loc_id.src_ = loc_id.src_ != source_idx_t::invalid()
                            ? loc_id.src_ + source_idx_offset
                            : source_idx_t::invalid();
          auto const loc_idx = i.second != location_idx_t::invalid()
                                   ? i.second + locations_offset
                                   : location_idx_t::invalid();
          auto const [it, is_new] =
              loc.location_id_to_idx_.emplace(loc_id, loc_idx);
          if (!is_new) {
            log(log_lvl::error, "loader.load", "duplicate station {}",
                loc_id.id_);
          }
        }
        for (auto const& i : new_locations.names_) {
          loc.names_.emplace_back(i);
        }
        for (auto const& i : new_locations.platform_codes_) {
          loc.platform_codes_.emplace_back(i);
        }
        for (auto const& i : new_locations.descriptions_) {
          loc.descriptions_.emplace_back(i);
        }
        for (auto const& i : new_locations.ids_) {
          loc.ids_.emplace_back(i);
        }
        for (auto const& i : new_locations.alt_names_) {
          auto vec = loc.alt_names_.add_back_sized(0U);
          for (auto const& j : i) {
            vec.push_back(j != alt_name_idx_t::invalid()
                              ? j + alt_name_idx_offset
                              : alt_name_idx_t::invalid());
          }
        }
        for (auto const& i : new_locations.coordinates_) {
          loc.coordinates_.push_back(i);
        }
        for (auto const& i : new_locations.src_) {
          loc.src_.push_back(i != source_idx_t::invalid()
                                 ? i + source_idx_offset
                                 : source_idx_t::invalid());
        }
        for (auto const& i : new_locations.transfer_time_) {
          loc.transfer_time_.push_back(i);
        }
        for (auto const& i : new_locations.types_) {
          loc.types_.push_back(i);
        }
        for (auto const& i : new_locations.parents_) {
          loc.parents_.push_back(i != location_idx_t::invalid()
                                     ? i + locations_offset
                                     : location_idx_t::invalid());
        }
        for (auto const& i : new_locations.location_timezones_) {
          loc.location_timezones_.push_back(i != timezone_idx_t::invalid()
                                                ? i + timezones_offset
                                                : timezone_idx_t::invalid());
        }
        for (auto const& i : new_locations.equivalences_) {
          auto entry = loc.equivalences_.emplace_back();
          for (auto const& j : i) {
            auto loc_idx = j != location_idx_t::invalid()
                               ? j + locations_offset
                               : location_idx_t::invalid();
            entry.emplace_back(loc_idx);
          }
        }
        for (auto const& i : new_locations.children_) {
          auto entry = loc.children_.emplace_back();
          for (auto const& j : i) {
            auto loc_idx = j != location_idx_t::invalid()
                               ? j + locations_offset
                               : location_idx_t::invalid();
            entry.emplace_back(loc_idx);
          }
        }
        for (auto const& i : new_locations.preprocessing_footpaths_out_) {
          auto entry = loc.preprocessing_footpaths_out_.emplace_back();
          for (auto const& j : i) {
            auto fp = footpath{j.target() != location_idx_t::invalid()
                                   ? j.target() + locations_offset
                                   : location_idx_t::invalid(),
                               j.duration()};
            entry.emplace_back(fp);
          }
        }
        for (auto const& i : new_locations.preprocessing_footpaths_in_) {
          auto entry = loc.preprocessing_footpaths_in_.emplace_back();
          for (auto const& j : i) {
            auto fp = footpath{j.target() != location_idx_t::invalid()
                                   ? j.target() + locations_offset
                                   : location_idx_t::invalid(),
                               j.duration()};
            entry.emplace_back(fp);
          }
        }
        /*
          loc.footpaths_out_ and loc.footpaths_in_ don't get used during loading
          and are thus skipped
        */
        assert(new_locations.footpaths_out_.size() == kNProfiles);
        for (auto const& i : new_locations.footpaths_out_) {
          assert(i.size() == 0);
        }
        assert(new_locations.footpaths_in_.size() == kNProfiles);
        for (auto const& i : new_locations.footpaths_in_) {
          assert(i.size() == 0);
        }
        for (auto const& i : new_locations.timezones_) {
          loc.timezones_.push_back(i);
        }
        /*
          loc.location_importance_ doesn't get used during loading and is thus
          skipped
        */
        assert(loc.location_importance_.size() == 0);
        for (auto const& i : new_locations.alt_name_strings_) {
          loc.alt_name_strings_.emplace_back(i);
        }
        for (auto const& i : new_locations.alt_name_langs_) {
          loc.alt_name_langs_.push_back(i != language_idx_t::invalid()
                                            ? i + language_offset
                                            : language_idx_t::invalid());
        }
        /*
          loc.max_importance_ and loc.rtree_ don't get used during loading
          and are thus skipped
        */
        assert(loc.max_importance_ == 0U);
      }  // end of locations struct
      for (auto const& i : new_location_routes) {
        auto vec = tt.location_routes_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != route_idx_t::invalid() ? j + route_idx_offset
                                                    : route_idx_t::invalid());
        }
      }
      auto const area_idx_offset = area_idx_t{tt.areas_.size()};
      for (auto const& i : new_location_areas) {
        auto vec = tt.location_areas_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != area_idx_t::invalid() ? j + area_idx_offset
                                                   : area_idx_t::invalid());
        }
      }
      for (location_idx_t i = location_idx_t{0};
           i < location_idx_t{new_location_location_groups.size()}; ++i) {
        tt.location_location_groups_.emplace_back_empty();
        for (auto const& j : new_location_location_groups[i]) {
          tt.location_location_groups_.back().push_back(
              j != location_group_idx_t::invalid()
                  ? j + location_group_offset
                  : location_group_idx_t::invalid());
        }
      }
      for (location_group_idx_t i = location_group_idx_t{0};
           i < location_group_idx_t{new_location_group_locations.size()}; ++i) {
        tt.location_group_locations_.emplace_back_empty();
        for (auto const& j :
             new_location_group_locations[location_group_idx_t{i}]) {
          tt.location_group_locations_.back().push_back(
              j != location_idx_t::invalid() ? j + locations_offset
                                             : location_idx_t::invalid());
        }
      }
      for (auto const& i : new_location_group_name) {
        tt.location_group_name_.emplace_back(string_map[i]);
      }
      for (auto const& i : new_location_group_id) {
        tt.location_group_id_.emplace_back(string_map[i]);
      }
      // tt.fwd_search_lb_graph_ not used during loading
      assert(tt.fwd_search_lb_graph_.size() == kNProfiles);
      for (auto const& i : tt.fwd_search_lb_graph_) {
        assert(i.size() == 0);
      }
      // tt.bwd_search_lb_graph_ not used during loading
      assert(tt.bwd_search_lb_graph_.size() == kNProfiles);
      for (auto const& i : tt.bwd_search_lb_graph_) {
        assert(i.size() == 0);
      }
      /*        route_idx_t	*/
      auto const transport_idx_offset =
          transport_idx_t{tt.transport_traffic_days_.size()};
      for (auto const& i : new_route_transport_ranges) {
        tt.route_transport_ranges_.push_back(interval{
            i.from_ != transport_idx_t::invalid()
                ? i.from_ + transport_idx_offset
                : transport_idx_t::invalid(),
            i.to_ != transport_idx_t::invalid() ? i.to_ + transport_idx_offset
                                                : transport_idx_t::invalid()});
      }
      for (auto const& i : new_route_location_seq) {
        auto vec = tt.route_location_seq_.add_back_sized(0U);
        for (auto const& j : i) {
          auto const s = stop{j};
          auto const mapped_location_idx =
              s.location_idx() != location_idx_t::invalid()
                  ? s.location_idx() + locations_offset
                  : location_idx_t::invalid();
          auto const mapped_stop =
              stop{mapped_location_idx, s.in_allowed_, s.out_allowed_,
                   s.in_allowed_wheelchair_, s.out_allowed_wheelchair_};
          vec.push_back(mapped_stop.value());
        }
      }
      for (auto const& i : new_route_clasz) {
        tt.route_clasz_.emplace_back(i);
      }
      for (auto const& i : new_route_section_clasz) {
        tt.route_section_clasz_.emplace_back(i);
      }
      for (auto const& i : new_route_bikes_allowed_per_section) {
        tt.route_bikes_allowed_per_section_.emplace_back(i);
      }
      for (auto const& i : new_route_cars_allowed_per_section) {
        tt.route_cars_allowed_per_section_.emplace_back(i);
      }
      auto const new_route_bikes_allowed_size = new_route_bikes_allowed.size();
      auto const route_bikes_allowed_size = tt.route_bikes_allowed_.size();
      tt.route_bikes_allowed_.resize(route_bikes_allowed_size +
                                     new_route_bikes_allowed_size);
      for (auto const& i : vw::iota(0U, new_route_bikes_allowed_size)) {
        tt.route_bikes_allowed_.set(route_bikes_allowed_size + i,
                                    new_route_bikes_allowed.test(i));
      }
      auto const new_route_cars_allowed_size = new_route_cars_allowed.size();
      auto const route_cars_allowed_size = tt.route_cars_allowed_.size();
      tt.route_cars_allowed_.resize(route_cars_allowed_size +
                                    new_route_cars_allowed_size);
      for (auto const& i : vw::iota(0U, new_route_cars_allowed_size)) {
        tt.route_cars_allowed_.set(route_cars_allowed_size + i,
                                   new_route_cars_allowed.test(i));
      }
      auto const route_stop_times_offset = tt.route_stop_times_.size();
      for (auto const& i : new_route_stop_time_ranges) {
        tt.route_stop_time_ranges_.push_back(
            interval{i.from_ + route_stop_times_offset,
                     i.to_ + route_stop_times_offset});
      }
      for (auto const& i : new_route_stop_times) {
        tt.route_stop_times_.push_back(i);
      }
      for (auto const& i : new_transport_route) {
        tt.transport_route_.push_back(i != route_idx_t::invalid()
                                          ? i + route_idx_offset
                                          : route_idx_t::invalid());
      }
      /*          fares		*/
      for (auto const& i : new_fares) {
        auto mapped_leg_group_name =
            vector_map<leg_group_idx_t, string_idx_t>{};
        for (auto const& j : i.leg_group_name_) {
          mapped_leg_group_name.push_back(string_map[j]);
        }
        auto mapped_fare_media =
            vector_map<fare_media_idx_t, fares::fare_media>{};
        for (auto const& j : i.fare_media_) {
          auto const mapped_media =
              fares::fare_media{.name_ = string_map[j.name_], .type_ = j.type_};
          mapped_fare_media.push_back(mapped_media);
        }
        auto mapped_fare_products =
            vecvec<fare_product_idx_t, fares::fare_product>{};
        for (auto const& j : i.fare_products_) {
          auto vec = mapped_fare_products.add_back_sized(0U);
          for (auto const& k : j) {
            auto const mapped_product = fares::fare_product{
                .amount_ = k.amount_,
                .name_ = string_map[k.name_],
                .media_ = k.media_,
                .currency_code_ = string_map[k.currency_code_],
                .rider_category_ = k.rider_category_};
            vec.push_back(mapped_product);
          }
        }
        auto mapped_fare_product_id =
            vector_map<fare_product_idx_t, string_idx_t>{};
        for (auto const& j : i.fare_product_id_) {
          mapped_fare_product_id.push_back(string_map[j]);
        }
        auto mapped_fare_leg_rules = vector<fares::fare_leg_rule>{};
        for (auto const& j : i.fare_leg_rules_) {
          auto const mapped_rule = fares::fare_leg_rule{
              .rule_priority_ = j.rule_priority_,
              .network_ = j.network_,
              .from_area_ = j.from_area_ != area_idx_t::invalid()
                                ? j.from_area_ + area_idx_offset
                                : area_idx_t::invalid(),
              .to_area_ = j.to_area_ != area_idx_t::invalid()
                              ? j.to_area_ + area_idx_offset
                              : area_idx_t::invalid(),
              .from_timeframe_group_ = j.from_timeframe_group_,
              .to_timeframe_group_ = j.to_timeframe_group_,
              .fare_product_ = j.fare_product_,
              .leg_group_idx_ = j.leg_group_idx_,
              .contains_exactly_area_set_id_ = j.contains_exactly_area_set_id_,
              .contains_area_set_id_ = j.contains_area_set_id_,
          };
          mapped_fare_leg_rules.push_back(mapped_rule);
        }
        auto mapped_fare_leg_join_rules = vector<fares::fare_leg_join_rule>{};
        for (auto const& j : i.fare_leg_join_rules_) {
          auto const mapped_join_rule = fares::fare_leg_join_rule{
              .from_network_ = j.from_network_,
              .to_network_ = j.to_network_,
              .from_stop_ = j.from_stop_ != location_idx_t::invalid()
                                ? j.from_stop_ + locations_offset
                                : location_idx_t::invalid(),
              .to_stop_ = j.to_stop_ != location_idx_t::invalid()
                              ? j.to_stop_ + locations_offset
                              : location_idx_t::invalid()};
          mapped_fare_leg_join_rules.push_back(mapped_join_rule);
        }
        auto mapped_rider_categories =
            vector_map<rider_category_idx_t, fares::rider_category>{};
        for (auto const& j : i.rider_categories_) {
          auto const mapped_rider_category = fares::rider_category{
              .name_ = string_map[j.name_],
              .eligibility_url_ = string_map[j.eligibility_url_],
              .is_default_fare_category_ = j.is_default_fare_category_};
          mapped_rider_categories.push_back(mapped_rider_category);
        }
        auto mapped_timeframes =
            vecvec<timeframe_group_idx_t, fares::timeframe>{};
        for (auto const& j : i.timeframes_) {
          auto vec = mapped_timeframes.add_back_sized(0U);
          for (auto const& k : j) {
            auto const mapped_timeframe =
                fares::timeframe{.start_time_ = k.start_time_,
                                 .end_time_ = k.end_time_,
                                 .service_ = k.service_,
                                 .service_id_ = string_map[k.service_id_]};
            vec.push_back(mapped_timeframe);
          }
        }
        auto mapped_timeframe_id =
            vector_map<timeframe_group_idx_t, string_idx_t>{};
        for (auto const& j : i.timeframe_id_) {
          mapped_timeframe_id.push_back(string_map[j]);
        }
        auto mapped_networks = vector_map<network_idx_t, fares::network>{};
        for (auto const& j : i.networks_) {
          auto const mapped_network = fares::network{
              .id_ = string_map[j.id_],
              .name_ = string_map[j.name_],
          };
          mapped_networks.push_back(mapped_network);
        }
        auto mapped_area_sets = vecvec<area_set_idx_t, area_idx_t>{};
        for (auto const& j : i.area_sets_) {
          auto vec = mapped_area_sets.add_back_sized(0U);
          for (auto const& k : j) {
            vec.push_back(k != area_idx_t::invalid() ? k + area_idx_offset
                                                     : area_idx_t::invalid());
          }
        }
        auto mapped_area_set_ids = vector_map<area_set_idx_t, string_idx_t>{};
        for (auto const& j : i.area_set_ids_) {
          mapped_area_set_ids.push_back(string_map[j]);
        }
        auto const mapped_fares =
            fares{.leg_group_name_ = mapped_leg_group_name,
                  .fare_media_ = mapped_fare_media,
                  .fare_products_ = mapped_fare_products,
                  .fare_product_id_ = mapped_fare_product_id,
                  .fare_leg_rules_ = mapped_fare_leg_rules,
                  .fare_leg_join_rules_ = mapped_fare_leg_join_rules,
                  .fare_transfer_rules_ = i.fare_transfer_rules_,
                  .rider_categories_ = mapped_rider_categories,
                  .timeframes_ = mapped_timeframes,
                  .timeframe_id_ = mapped_timeframe_id,
                  .route_networks_ = i.route_networks_,
                  .networks_ = mapped_networks,
                  .area_sets_ = mapped_area_sets,
                  .area_set_ids_ = mapped_area_set_ids,
                  .has_priority_ = i.has_priority_};
        tt.fares_.push_back(mapped_fares);
      }
      /*      provider_idx_t	*/
      auto const provider_idx_offset = provider_idx_t{tt.providers_.size()};
      for (auto const& i : new_providers) {
        auto const p = provider{.id_ = string_map[i.id_],
                                .name_ = string_map[i.name_],
                                .url_ = string_map[i.url_],
                                .tz_ = i.tz_ != timezone_idx_t::invalid()
                                           ? i.tz_ + timezones_offset
                                           : timezone_idx_t::invalid(),
                                .src_ = i.src_ != source_idx_t::invalid()
                                            ? i.src_ + source_idx_offset
                                            : source_idx_t::invalid()};
        tt.providers_.push_back(p);
      }
      for (auto const& i : new_provider_id_to_idx) {
        tt.provider_id_to_idx_.push_back(i != provider_idx_t::invalid()
                                             ? i + provider_idx_offset
                                             : provider_idx_t::invalid());
      }
      /*	  Flex		*/
      for (auto const& i : new_flex_area_bbox) {
        tt.flex_area_bbox_.push_back(i);
      }
      for (auto const& i : new_flex_area_id) {
        tt.flex_area_id_.push_back(string_map[i]);
      }
      for (auto const& i : new_flex_area_src) {
        tt.flex_area_src_.push_back(i != source_idx_t::invalid()
                                        ? i + source_idx_offset
                                        : source_idx_t::invalid());
      }
      // tt.flex_area_locations_ not used during loading
      assert(tt.flex_area_locations_.size() == 0);
      for (auto const& i : new_flex_area_outers) {
        tt.flex_area_outers_.emplace_back(i);
      }
      for (auto const& i : new_flex_area_inners) {
        tt.flex_area_inners_.emplace_back(i);
      }
      for (auto const& i : new_flex_area_name) {
        tt.flex_area_name_.emplace_back(i);
      }
      for (auto const& i : new_flex_area_desc) {
        tt.flex_area_desc_.emplace_back(i);
      }
      for (auto const& n : new_flex_area_rtree.nodes_) {
        if (n.kind_ == rtree<flex_area_idx_t>::kind::kLeaf) {
          for (size_t i = 0; i < n.count_; ++i) {
            tt.flex_area_rtree_.insert(n.rects_[i].min_, n.rects_[i].max_,
                                       n.data_[i]);
          }
        }
      }
      auto const flex_transport_traffic_days_offset =
          flex_transport_idx_t{tt.flex_transport_traffic_days_.size()};
      for (location_group_idx_t i = location_group_idx_t{0};
           i < location_group_idx_t{new_location_group_transports.size()};
           ++i) {
        tt.location_group_transports_.emplace_back_empty();
        for (auto const& j : new_location_group_transports[i]) {
          tt.location_group_transports_.back().push_back(
              j != flex_transport_idx_t::invalid()
                  ? j + flex_transport_traffic_days_offset
                  : flex_transport_idx_t::invalid());
        }
      }
      for (flex_area_idx_t i = flex_area_idx_t{0};
           i < flex_area_idx_t{new_flex_area_transports.size()}; ++i) {
        tt.flex_area_transports_.emplace_back_empty();
        for (auto const& j : new_flex_area_transports[i]) {
          tt.flex_area_transports_.back().push_back(
              j != flex_transport_idx_t::invalid()
                  ? j + flex_transport_traffic_days_offset
                  : flex_transport_idx_t::invalid());
        }
      }
      for (auto const& i : new_flex_transport_traffic_days) {
        tt.flex_transport_traffic_days_.push_back(
            i != bitfield_idx_t::invalid() ? corrected_indices[i]
                                           : bitfield_idx_t::invalid());
      }
      for (auto const& i : new_flex_transport_trip) {
        tt.flex_transport_trip_.push_back(i != trip_idx_t::invalid()
                                              ? i + trip_offset
                                              : trip_idx_t::invalid());
      }
      for (auto const& i : new_flex_transport_stop_time_windows) {
        tt.flex_transport_stop_time_windows_.emplace_back(i);
      }
      auto flex_stop_seq_offset = flex_stop_seq_idx_t{tt.flex_stop_seq_.size()};
      for (auto const& i : new_flex_transport_stop_seq) {
        tt.flex_transport_stop_seq_.push_back(
            i != flex_stop_seq_idx_t::invalid()
                ? i + flex_stop_seq_offset
                : flex_stop_seq_idx_t::invalid());
      }
      for (auto const& i : new_flex_stop_seq) {
        tt.flex_stop_seq_.emplace_back(i);
      }
      auto booking_rules_offset = booking_rule_idx_t{tt.booking_rules_.size()};
      for (auto const& i : new_flex_transport_pickup_booking_rule) {
        auto vec = tt.flex_transport_pickup_booking_rule_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != booking_rule_idx_t::invalid()
                            ? j + booking_rules_offset
                            : booking_rule_idx_t::invalid());
        }
      }
      for (auto const& i : new_flex_transport_drop_off_booking_rule) {
        auto vec = tt.flex_transport_drop_off_booking_rule_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != booking_rule_idx_t::invalid()
                            ? j + booking_rules_offset
                            : booking_rule_idx_t::invalid());
        }
      }
      for (auto const& i : new_booking_rules) {
        auto const b =
            booking_rule{.id_ = string_map[i.id_],
                         .type_ = i.type_,
                         .message_ = string_map[i.message_],
                         .pickup_message_ = string_map[i.pickup_message_],
                         .drop_off_message_ = string_map[i.drop_off_message_],
                         .phone_number_ = string_map[i.phone_number_],
                         .info_url_ = string_map[i.info_url_],
                         .booking_url_ = string_map[i.booking_url_]};
        tt.booking_rules_.push_back(b);
      }
      /*      trip_id_idx_t	*/
      auto trip_id_offset = trip_id_idx_t{tt.trip_id_strings_.size()};
      for (auto const& i : new_trip_id_to_idx) {
        tt.trip_id_to_idx_.push_back(pair<trip_id_idx_t, trip_idx_t>{
            i.first != trip_id_idx_t::invalid() ? i.first + trip_id_offset
                                                : trip_id_idx_t::invalid(),
            i.second != trip_idx_t::invalid() ? i.second + trip_offset
                                              : trip_idx_t::invalid()});
      }
      for (auto const& i : new_trip_ids) {
        auto entry = tt.trip_ids_.emplace_back();
        for (auto const& j : i) {
          auto trip_id = trip_id_idx_t{j != trip_id_idx_t::invalid()
                                           ? j + trip_id_offset
                                           : trip_id_idx_t::invalid()};
          entry.emplace_back(trip_id);
        }
      }
      for (auto const& i : new_trip_id_src) {
        tt.trip_id_src_.push_back(i != source_idx_t::invalid()
                                      ? i + source_idx_offset
                                      : source_idx_t::invalid());
      }
      for (auto const& i : new_trip_id_strings) {
        tt.trip_id_strings_.emplace_back(i);
      }
      // tt.trip_train_nr_ not used during loading
      assert(tt.trip_train_nr_.size() == 0);
      /* 	 trip_idx_t	 */
      auto const add_size = trip_idx_t{new_trip_direction_id.size()};
      tt.trip_direction_id_.resize(to_idx(trip_offset + add_size));
      for (auto const& i : vw::iota(0U, to_idx(add_size))) {
        auto const idx = trip_idx_t{i};
        tt.trip_direction_id_.set(idx + trip_offset,
                                  new_trip_direction_id.test(idx));
      }
      for (auto const& i : new_trip_route_id) {
        tt.trip_route_id_.push_back(i);
      }
      for (auto i = trip_idx_t{0};
           i < trip_idx_t{new_trip_transport_ranges.size()}; ++i) {
        tt.trip_transport_ranges_.emplace_back(new_trip_transport_ranges[i]);
      }
      for (auto const& i : new_trip_stop_seq_numbers) {
        tt.trip_stop_seq_numbers_.emplace_back(i);
      }
      for (auto const& i : new_trip_short_names) {
        tt.trip_short_names_.emplace_back(i);
      }
      for (auto const& i : new_trip_display_names) {
        tt.trip_display_names_.emplace_back(i);
      }
      auto const merged_trips_idx_offset =
          merged_trips_idx_t{tt.merged_trips_.size()};
      for (auto const& i : new_merged_trips) {
        auto vec = tt.merged_trips_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != trip_idx_t::invalid() ? j + trip_offset
                                                   : trip_idx_t::invalid());
        }
      }
      /*      route_id_idx_t	 */
      for (auto const& i : new_route_ids) {
        auto vec = paged_vecvec<route_id_idx_t, trip_idx_t>{};
        for (route_id_idx_t j = route_id_idx_t{0};
             j < route_id_idx_t{i.route_id_trips_.size()}; ++j) {
          vec.emplace_back_empty();
          for (auto const& k : i.route_id_trips_[j]) {
            vec.back().push_back(k + trip_offset);
          }
        }
        auto mapped_providers = vector_map<route_id_idx_t, provider_idx_t>{};
        for (auto const& j : i.route_id_provider_) {
          mapped_providers.push_back(j != provider_idx_t::invalid()
                                         ? j + provider_idx_offset
                                         : provider_idx_t::invalid());
        }
        auto const mapped_route_ids = timetable::route_ids{
            .route_id_short_names_ = i.route_id_short_names_,
            .route_id_long_names_ = i.route_id_long_names_,
            .route_id_type_ = i.route_id_type_,
            .route_id_provider_ = mapped_providers,
            .route_id_colors_ = i.route_id_colors_,
            .route_id_trips_ = vec,
            .ids_ = i.ids_};
        tt.route_ids_.push_back(mapped_route_ids);
      }
      /*     transport_idx_t	*/
      for (auto const& i : new_transport_first_dep_offset) {
        tt.transport_first_dep_offset_.push_back(i);
      }
      // tt.initial_day_offset_ not used during loading
      assert(tt.initial_day_offset_.size() == 0);
      for (auto const& i : new_transport_traffic_days) {
        tt.transport_traffic_days_.push_back(i != bitfield_idx_t::invalid()
                                                 ? corrected_indices[i]
                                                 : bitfield_idx_t::invalid());
      }
      for (auto const& i : new_transport_to_trip_section) {
        auto vec = tt.transport_to_trip_section_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != merged_trips_idx_t::invalid()
                            ? j + merged_trips_idx_offset
                            : merged_trips_idx_t::invalid());
        }
      }
      for (auto const& i : new_transport_section_attributes) {
        tt.transport_section_attributes_.emplace_back(i);
      }
      for (auto const& i : new_transport_section_providers) {
        auto vec = tt.transport_section_providers_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != provider_idx_t::invalid()
                            ? j + provider_idx_offset
                            : provider_idx_t::invalid());
        }
      }
      for (auto const& i : new_transport_section_directions) {
        tt.transport_section_directions_.emplace_back(i);
      }
      auto const trip_lines_offset = trip_line_idx_t{tt.trip_lines_.size()};
      for (auto const& i : new_transport_section_lines) {
        auto vec = tt.transport_section_lines_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != trip_line_idx_t::invalid()
                            ? j + trip_lines_offset
                            : trip_line_idx_t::invalid());
        }
      }
      for (auto const& i : new_transport_section_route_colors) {
        tt.transport_section_route_colors_.emplace_back(i);
      }
      /*        Meta infos	*/
      for (auto const& i : new_trip_lines) {
        tt.trip_lines_.emplace_back(i);
      }
      /*        area_idx_t	*/
      for (auto const& i : new_areas) {
        tt.areas_.push_back(area{string_map[i.id_], string_map[i.name_]});
      }
      /*      attribute_idx_t	*/
      auto const attribute_idx_offset = attribute_idx_t{tt.attributes_.size()};
      for (auto const& i : new_attributes) {
        tt.attributes_.push_back(i);
      }
      for (auto const& i : new_attribute_combinations) {
        auto vec = tt.attribute_combinations_.add_back_sized(0U);
        for (auto const& j : i) {
          vec.push_back(j != attribute_idx_t::invalid()
                            ? j + attribute_idx_offset
                            : attribute_idx_t::invalid());
        }
      }
      /*  trip_direction_string_idx_t	*/
      for (auto const& i : new_trip_direction_strings) {
        tt.trip_direction_strings_.emplace_back(i);
      }
      for (auto const& i : new_trip_directions) {
        tt.trip_directions_.push_back(im.map(i));
      }
      /*     Other	*/
      tt.n_sources_ += new_n_sources;
      assert(tt.n_sources_ == tt.src_end_date_.size());
      // tt.profiles_ not used during loading
      assert(tt.profiles_.size() == 0);
      // tt.date_range_ not changed
      assert(tt.date_range_ == date_range);
      /* Save snapshot */
      fs::create_directories(local_cache_path);
      if (shapes != nullptr) {
        shapes->add(shape_store.get());
        shape_store =
            std::make_unique<shapes_storage>(local_cache_path, shapes->mode_);
        shape_store->add(shapes);
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

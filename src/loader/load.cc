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
  alt_name_idx_t const alt_name_idx_offset_;
  language_idx_t const language_idx_offset_;
  location_group_idx_t const location_group_idx_offset_;
  location_idx_t const location_idx_offset_;
  source_file_idx_t const source_file_idx_offset_;
  trip_direction_string_idx_t const trip_direction_string_idx_offset_;

  index_mapping(timetable const& first_tt)
      : alt_name_idx_offset_{first_tt.locations_.alt_name_strings_.size()},
        language_idx_offset_{first_tt.languages_.size()},
        location_group_idx_offset_{first_tt.location_group_name_.size()},
        location_idx_offset_{first_tt.n_locations()},
        source_file_idx_offset_{first_tt.source_file_names_.size()},
        trip_direction_string_idx_offset_{
            first_tt.trip_direction_strings_.size()} {}

  auto map(alt_name_idx_t const& i) const {
    return i != alt_name_idx_t::invalid() ? i + alt_name_idx_offset_
                                          : alt_name_idx_t::invalid();
  }
  auto map(language_idx_t const& i) const {
    return i != language_idx_t::invalid() ? i + language_idx_offset_
                                          : language_idx_t::invalid();
  }
  auto map(location_group_idx_t const& i) const {
    return i != location_group_idx_t::invalid()
               ? i + location_group_idx_offset_
               : location_group_idx_t::invalid();
  }
  auto map(location_idx_t const& i) const {
    return i != location_idx_t::invalid() ? i + location_idx_offset_
                                          : location_idx_t::invalid();
  }
  auto map(source_file_idx_t const& i) const {
    return i != source_file_idx_t::invalid() ? i + source_file_idx_offset_
                                             : source_file_idx_t::invalid();
  }
  auto map(stop const& i) const {
    return stop{map(i.location_idx()), i.in_allowed_, i.out_allowed_,
                i.in_allowed_wheelchair_, i.out_allowed_wheelchair_};
  }
  auto map(trip_debug const& i) const {
    return trip_debug{map(i.source_file_idx_), i.line_number_from_,
                      i.line_number_to_};
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

  auto map(fares::fare_leg_join_rule const& i) const {
    return fares::fare_leg_join_rule{i.from_network_, i.to_network_,
                                     map(i.from_stop_), map(i.to_stop_)};
  }
  auto map(footpath const& i) const {
    return footpath{map(i.target()), i.duration()};
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
      auto other_tt = result.first;
      auto shape_store = std::move(result.second);
      /* Save new data */
      auto const new_bitfields = other_tt.bitfields_;
      auto const new_source_end_date = other_tt.src_end_date_;
      auto const new_trip_id_to_idx = other_tt.trip_id_to_idx_;
      auto const new_trip_ids = other_tt.trip_ids_;
      auto const new_trip_id_strings = other_tt.trip_id_strings_;
      auto const new_trip_id_src = other_tt.trip_id_src_;
      auto const new_trip_direction_id = other_tt.trip_direction_id_;
      auto const new_trip_route_id = other_tt.trip_route_id_;
      auto const new_route_ids = other_tt.route_ids_;
      auto const new_trip_transport_ranges = other_tt.trip_transport_ranges_;
      auto const new_trip_stop_seq_numbers = other_tt.trip_stop_seq_numbers_;
      auto const new_source_file_names = other_tt.source_file_names_;
      auto const new_trip_debug = other_tt.trip_debug_;
      auto const new_trip_short_names = other_tt.trip_short_names_;
      auto const new_trip_display_names = other_tt.trip_display_names_;
      auto const new_route_transport_ranges = other_tt.route_transport_ranges_;
      auto const new_route_location_seq = other_tt.route_location_seq_;
      auto const new_route_clasz = other_tt.route_clasz_;
      auto const new_route_section_clasz = other_tt.route_section_clasz_;
      auto const new_route_bikes_allowed = other_tt.route_bikes_allowed_;
      auto const new_route_cars_allowed = other_tt.route_cars_allowed_;
      auto const new_route_bikes_allowed_per_section =
          other_tt.route_bikes_allowed_per_section_;
      auto const new_route_cars_allowed_per_section =
          other_tt.route_cars_allowed_per_section_;
      auto const new_route_stop_time_ranges = other_tt.route_stop_time_ranges_;
      auto const new_route_stop_times = other_tt.route_stop_times_;
      auto const new_transport_first_dep_offset =
          other_tt.transport_first_dep_offset_;
      auto const new_transport_traffic_days = other_tt.transport_traffic_days_;
      auto const new_transport_route = other_tt.transport_route_;
      auto const new_transport_to_trip_section =
          other_tt.transport_to_trip_section_;
      auto const new_languages = other_tt.languages_;
      auto const new_locations = other_tt.locations_;
      auto const new_merged_trips = other_tt.merged_trips_;
      auto const new_attributes = other_tt.attributes_;
      auto const new_attribute_combinations = other_tt.attribute_combinations_;
      auto const new_trip_direction_strings = other_tt.trip_direction_strings_;
      auto const new_trip_directions = other_tt.trip_directions_;
      auto const new_trip_lines = other_tt.trip_lines_;
      auto const new_transport_section_attributes =
          other_tt.transport_section_attributes_;
      auto const new_transport_section_providers =
          other_tt.transport_section_providers_;
      auto const new_transport_section_directions =
          other_tt.transport_section_directions_;
      auto const new_transport_section_lines =
          other_tt.transport_section_lines_;
      auto const new_transport_section_route_colors =
          other_tt.transport_section_route_colors_;
      auto const new_location_routes = other_tt.location_routes_;
      auto const new_providers = other_tt.providers_;
      auto const new_provider_id_to_idx = other_tt.provider_id_to_idx_;
      auto const new_fares = other_tt.fares_;
      auto const new_areas = other_tt.areas_;
      auto const new_location_areas = other_tt.location_areas_;
      auto const new_location_location_groups =
          other_tt.location_location_groups_;
      auto const new_location_group_locations =
          other_tt.location_group_locations_;
      auto const new_location_group_name = other_tt.location_group_name_;
      auto const new_location_group_id = other_tt.location_group_id_;
      auto const new_flex_area_bbox = other_tt.flex_area_bbox_;
      auto const new_flex_area_id = other_tt.flex_area_id_;
      auto const new_flex_area_src = other_tt.flex_area_src_;
      auto const new_flex_area_outers = other_tt.flex_area_outers_;
      auto const new_flex_area_inners = other_tt.flex_area_inners_;
      auto const new_flex_area_name = other_tt.flex_area_name_;
      auto const new_flex_area_desc = other_tt.flex_area_desc_;
      auto const new_flex_area_rtree = other_tt.flex_area_rtree_;
      auto const new_location_group_transports =
          other_tt.location_group_transports_;
      auto const new_flex_area_transports = other_tt.flex_area_transports_;
      auto const new_flex_transport_traffic_days =
          other_tt.flex_transport_traffic_days_;
      auto const new_flex_transport_trip = other_tt.flex_transport_trip_;
      auto const new_flex_transport_stop_time_windows =
          other_tt.flex_transport_stop_time_windows_;
      auto const new_flex_transport_stop_seq =
          other_tt.flex_transport_stop_seq_;
      auto const new_flex_stop_seq = other_tt.flex_stop_seq_;
      auto const new_flex_transport_pickup_booking_rule =
          other_tt.flex_transport_pickup_booking_rule_;
      auto const new_flex_transport_drop_off_booking_rule =
          other_tt.flex_transport_drop_off_booking_rule_;
      auto const new_booking_rules = other_tt.booking_rules_;
      auto const new_strings = other_tt.strings_;
      auto const new_n_sources = other_tt.n_sources_;
      progress_tracker->status("Saved new data");
      /* Add new data and adjust references */
      auto const im = index_mapping(tt);
      /*	bitfields	*/
      auto corrected_indices = vector_map<bitfield_idx_t, bitfield_idx_t>{};
      auto bitfields_ = hash_map<bitfield, bitfield_idx_t>{};
      for (auto const [idx_, bf] : utl::enumerate(tt.bitfields_)) {
        auto new_idx =
            utl::get_or_create(bitfields_, bf, [&]() { return idx_; });
        assert(new_idx == idx_);  // bitfields must be unique in the timetable
      }
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
      for (auto const& i : new_source_file_names) {
        tt.source_file_names_.emplace_back(i);
      }
      for (auto const& i : new_trip_debug) {
        auto entry = tt.trip_debug_.emplace_back();
        for (auto const& j : i) {
          entry.emplace_back(im.map(j));
        }
      }
      /*	 languages	*/
      for (auto const& i : new_languages) {
        tt.languages_.emplace_back(i);
      }
      /*       location_idx_t	*/
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
          auto const loc_idx = im.map(i.second);
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
            vec.push_back(im.map(j));
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
          loc.parents_.push_back(im.map(i));
        }
        for (auto const& i : new_locations.location_timezones_) {
          loc.location_timezones_.push_back(i != timezone_idx_t::invalid()
                                                ? i + timezones_offset
                                                : timezone_idx_t::invalid());
        }
        for (auto const& i : new_locations.equivalences_) {
          auto entry = loc.equivalences_.emplace_back();
          for (auto const& j : i) {
            entry.emplace_back(im.map(j));
          }
        }
        for (auto const& i : new_locations.children_) {
          auto entry = loc.children_.emplace_back();
          for (auto const& j : i) {
            entry.emplace_back(im.map(j));
          }
        }
        for (auto const& i : new_locations.preprocessing_footpaths_out_) {
          auto entry = loc.preprocessing_footpaths_out_.emplace_back();
          for (auto const& j : i) {
            entry.emplace_back(im.map(j));
          }
        }
        for (auto const& i : new_locations.preprocessing_footpaths_in_) {
          auto entry = loc.preprocessing_footpaths_in_.emplace_back();
          for (auto const& j : i) {
            entry.emplace_back(im.map(j));
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
          loc.alt_name_langs_.push_back(im.map(i));
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
          tt.location_location_groups_.back().push_back(im.map(j));
        }
      }
      for (location_group_idx_t i = location_group_idx_t{0};
           i < location_group_idx_t{new_location_group_locations.size()}; ++i) {
        tt.location_group_locations_.emplace_back_empty();
        for (auto const& j :
             new_location_group_locations[location_group_idx_t{i}]) {
          tt.location_group_locations_.back().push_back(im.map(j));
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
          vec.push_back(im.map(stop{j}).value());
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
          mapped_fare_leg_join_rules.push_back(im.map(j));
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

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
      tt.n_sources_ = static_cast<cista::base_t<source_idx_t>>(sources.size());
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
      auto const old_transport_traffic_days_ = tt.transport_traffic_days_;
      tt.transport_traffic_days_ = old_transport_traffic_days_;
      auto const old_flex_transport_traffic_days_ =
          tt.flex_transport_traffic_days_;
      tt.flex_transport_traffic_days_ = old_flex_transport_traffic_days_;
      auto const old_source_end_date = tt.src_end_date_;
      tt.src_end_date_ = old_source_end_date;
      auto const old_source_file_names = tt.source_file_names_;
      tt.source_file_names_ = old_source_file_names;
      auto const old_trip_debug = tt.trip_debug_;
      tt.trip_debug_ = old_trip_debug;
      auto const old_route_location_seq = tt.route_location_seq_;
      tt.route_location_seq_ = old_route_location_seq;
      auto const old_languages = tt.languages_;
      tt.languages_ = old_languages;
      auto const old_locations = tt.locations_;
      tt.locations_ = old_locations;
      auto const old_location_routes = tt.location_routes_;
      tt.location_routes_ = old_location_routes;
      auto const old_providers = tt.providers_;
      tt.providers_ = old_providers;
      auto const old_fares = tt.fares_;
      tt.fares_ = old_fares;
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
      /* Prepare timetable by emptying corrected fields */
      tt.bitfields_.reset();
      auto bitfields = hash_map<bitfield, bitfield_idx_t>{};
      tt.src_end_date_.reset();
      tt.source_file_names_.clear();
      tt.trip_debug_ = mutable_fws_multimap<trip_idx_t, trip_debug>{};
      tt.languages_.clear();
      tt.locations_ = timetable::locations{};
      tt.location_routes_.clear();
      tt.location_areas_.clear();
      tt.location_group_locations_.clear();
      tt.location_location_groups_.clear();
      tt.location_group_name_.reset();
      tt.location_group_id_.reset();
      tt.flex_area_bbox_.reset();
      tt.flex_area_id_.reset();
      tt.flex_area_src_.reset();
      tt.flex_area_outers_ = nvec<flex_area_idx_t, geo::latlng, 2U>{};
      tt.flex_area_inners_ = nvec<flex_area_idx_t, geo::latlng, 3U>{};
      tt.flex_area_name_.clear();
      tt.flex_area_desc_.clear();
      tt.flex_area_rtree_ = rtree<flex_area_idx_t>{};
      tt.location_group_transports_.clear();
      tt.flex_area_transports_.clear();
      tt.flex_transport_trip_.reset();
      tt.flex_transport_stop_time_windows_.clear();
      tt.flex_transport_stop_seq_.reset();
      tt.flex_stop_seq_.clear();
      tt.flex_transport_pickup_booking_rule_.clear();
      tt.flex_transport_drop_off_booking_rule_.clear();
      tt.booking_rules_.reset();
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
      /* Load file */
      try {
        (*it)->load(local_config, src, *dir, tt, bitfields, a, shapes);
      } catch (std::exception const& e) {
        throw utl::fail("failed to load {}: {}", path, e.what());
      }
      /* Save new data */
      auto const new_bitfields = tt.bitfields_;
      auto new_transport_traffic_days_ =
          vector_map<transport_idx_t, bitfield_idx_t>{};
      for (auto i = old_transport_traffic_days_.size();
           i < tt.transport_traffic_days_.size(); ++i) {
        new_transport_traffic_days_.push_back(
            tt.transport_traffic_days_[transport_idx_t{i}]);
      }
      auto new_flex_transport_traffic_days_ =
          vector_map<flex_transport_idx_t, bitfield_idx_t>{};
      for (auto i = old_flex_transport_traffic_days_.size();
           i < tt.flex_transport_traffic_days_.size(); ++i) {
        new_flex_transport_traffic_days_.push_back(
            tt.flex_transport_traffic_days_[flex_transport_idx_t{i}]);
      }
      auto const new_source_end_date = tt.src_end_date_;
      auto const new_source_file_names = tt.source_file_names_;
      auto const new_trip_debug = tt.trip_debug_;
      auto new_route_location_seq = vecvec<route_idx_t, stop::value_type>{};
      for (auto idx = old_route_location_seq.size();
           idx < tt.route_location_seq_.size(); ++idx) {
        auto vec = new_route_location_seq.add_back_sized(0U);
        for (auto const& j : tt.route_location_seq_[route_idx_t{idx}]) {
          vec.push_back(j);
        }
      }
      auto const new_languages = tt.languages_;
      auto const new_locations = tt.locations_;
      auto const new_location_routes = tt.location_routes_;
      auto new_providers = vector_map<provider_idx_t, provider>{};
      for (auto i = old_providers.size(); i < tt.providers_.size(); ++i) {
        new_providers.push_back(tt.providers_[provider_idx_t{i}]);
      }
      // last fares are new
      auto new_fares = vector_map<source_idx_t, fares>{};
      new_fares.push_back(tt.fares_[src]);
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
      progress_tracker->status("Saved new data");
      /* Restore old timetable */
      tt.bitfields_ = old_bitfields;
      tt.transport_traffic_days_ = old_transport_traffic_days_;
      tt.flex_transport_traffic_days_ = old_flex_transport_traffic_days_;
      tt.src_end_date_ = old_source_end_date;
      tt.source_file_names_ = old_source_file_names;
      tt.trip_debug_ = old_trip_debug;
      tt.route_location_seq_ = old_route_location_seq;
      tt.languages_ = old_languages;
      tt.locations_ = old_locations;
      tt.location_routes_ = old_location_routes;
      tt.providers_ = old_providers;
      tt.fares_ = old_fares;
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
      /* Add new data and adjust references */
      /*	bitfields	*/
      auto corrected_indices = vector_map<bitfield_idx_t, bitfield_idx_t>{};
      for (auto const& [idx_, bf] : utl::enumerate(new_bitfields)) {
        auto adjusted_idx = utl::get_or_create(
            bitfields_, bf, [&]() { return tt.register_bitfield(bf); });
        corrected_indices.emplace_back(adjusted_idx);
      }
      for (auto const& i : new_transport_traffic_days_) {
        tt.transport_traffic_days_.push_back(corrected_indices[i]);
      }
      for (auto const& i : new_flex_transport_traffic_days_) {
        tt.flex_transport_traffic_days_.push_back(corrected_indices[i]);
      }
      /*	 sources	*/
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
      {  // merge locations struct
        auto&& loc = tt.locations_;
        for (auto const& i : new_locations.location_id_to_idx_) {
          auto const loc_id = i.first;
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
          loc.src_.push_back(i);
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
        tt.location_routes_.emplace_back(i);
      }
      for (auto const& i : new_location_areas) {
        tt.location_areas_.emplace_back(i);
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
        tt.location_group_name_.emplace_back(i);
      }
      for (auto const& i : new_location_group_id) {
        tt.location_group_id_.emplace_back(i);
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
      /*          fares		*/
      for (auto const& i : new_fares) {
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
        auto const mapped_fares =
            fares{.leg_group_name_ = i.leg_group_name_,
                  .fare_media_ = i.fare_media_,
                  .fare_products_ = i.fare_products_,
                  .fare_product_id_ = i.fare_product_id_,
                  .fare_leg_rules_ = i.fare_leg_rules_,
                  .fare_leg_join_rules_ = mapped_fare_leg_join_rules,
                  .fare_transfer_rules_ = i.fare_transfer_rules_,
                  .rider_categories_ = i.rider_categories_,
                  .timeframes_ = i.timeframes_,
                  .timeframe_id_ = i.timeframe_id_,
                  .route_networks_ = i.route_networks_,
                  .networks_ = i.networks_,
                  .area_sets_ = i.area_sets_,
                  .area_set_ids_ = i.area_set_ids_,
                  .has_priority_ = i.has_priority_};
        tt.fares_.push_back(mapped_fares);
      }
      /*      provider_idx_t	*/
      for (auto const& i : new_providers) {
        auto const p = provider{.id_ = i.id_,
                                .name_ = i.name_,
                                .url_ = i.url_,
                                .tz_ = i.tz_ != timezone_idx_t::invalid()
                                           ? i.tz_ + timezones_offset
                                           : timezone_idx_t::invalid(),
                                .src_ = i.src_};
        tt.providers_.push_back(p);
      }
      /*	  Flex		*/
      for (auto const& i : new_flex_area_bbox) {
        tt.flex_area_bbox_.push_back(i);
      }
      for (auto const& i : new_flex_area_id) {
        tt.flex_area_id_.push_back(i);
      }
      for (auto const& i : new_flex_area_src) {
        tt.flex_area_src_.push_back(i);
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
      for (location_group_idx_t i = location_group_idx_t{0};
           i < location_group_idx_t{new_location_group_transports.size()};
           ++i) {
        tt.location_group_transports_.emplace_back_empty();
        for (auto const& j : new_location_group_transports[i]) {
          tt.location_group_transports_.back().push_back(j);
        }
      }
      for (flex_area_idx_t i = flex_area_idx_t{0};
           i < flex_area_idx_t{new_flex_area_transports.size()}; ++i) {
        tt.flex_area_transports_.emplace_back_empty();
        for (auto const& j : new_flex_area_transports[i]) {
          tt.flex_area_transports_.back().push_back(j);
        }
      }
      for (auto const& i : new_flex_transport_trip) {
        tt.flex_transport_trip_.push_back(i);
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
        tt.booking_rules_.push_back(i);
      }
      /* Save snapshot */
      fs::create_directories(local_cache_path);
      if (shapes != nullptr) {
        auto shape_store =
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

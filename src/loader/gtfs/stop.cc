#include "nigiri/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "geo/point_rtree.h"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

struct stop {
  void compute_close_stations(geo::point_rtree const& stop_rtree,
                              unsigned const link_stop_distance) {
    if (std::abs(coord_.lat_) < 2.0 && std::abs(coord_.lng_) < 2.0) {
      return;
    }
    close_ = utl::to_vec(
        stop_rtree.in_radius(coord_, link_stop_distance),
        [](std::size_t const idx) { return static_cast<unsigned>(idx); });
  }

  hash_set<stop*>& get_metas(std::vector<stop*> const& stops,
                             hash_set<stop*>& todo,
                             hash_set<stop*>& done) {
    todo.clear();
    done.clear();

    todo.emplace(this);
    todo.insert(begin(same_name_), end(same_name_));
    for (auto const& idx : close_) {
      todo.insert(stops[idx]);
    }

    while (!todo.empty()) {
      auto const next = *todo.begin();
      todo.erase(todo.begin());
      done.emplace(next);

      if (next->parent_ != nullptr && done.find(next->parent_) == end(done)) {
        todo.emplace(next->parent_);
      }

      for (auto const& p : next->children_) {
        if (done.find(p) == end(done)) {
          todo.emplace(p);
        }
      }
    }

    for (auto it = begin(done); it != end(done);) {
      auto* meta = *it;
      auto const is_parent = parent_ == meta;
      auto const is_child = children_.find(meta) != end(children_);
      auto const distance_in_m = geo::distance(meta->coord_, coord_);
      if ((distance_in_m > 500 && !is_parent && !is_child) ||
          distance_in_m > 2000) {
        it = done.erase(it);
      } else {
        ++it;
      }
    }

    for (auto const& d : done) {
      for (auto const& c : d->children_) {
        done.insert(c);
      }
    }

    return done;
  }

  std::string_view id_;
  std::string_view name_;
  std::string_view platform_code_;
  geo::latlng coord_;
  std::string_view timezone_;
  std::set<stop*> same_name_, children_;
  stop* parent_{nullptr};
  std::vector<unsigned> close_;
  location_idx_t location_{location_idx_t::invalid()};
  std::vector<footpath> footpaths_;
};

using stop_map_t = hash_map<std::string_view, std::unique_ptr<stop>>;

enum class transfer_type : std::uint8_t {
  kRecommended = 0U,
  kTimed = 1U,
  kMinimumChangeTime = 2U,
  kNotPossible = 3U,
  kStaySeated = 4U,
  kNoStaySeated = 5U,
  kGenerated = std::numeric_limits<std::uint8_t>::max()
};

void read_transfers(stop_map_t& stops, std::string_view file_content) {
  auto const timer = scoped_timer{"gtfs.loader.stops.transfers"};

  struct csv_transfer {
    utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_stop_id_;
    utl::csv_col<int, UTL_NAME("transfer_type")> transfer_type_;
    utl::csv_col<int, UTL_NAME("min_transfer_time")> min_transfer_time_;
  };

  if (file_content.empty()) {
    return;
  }

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Transfers")
      .out_bounds(15.F, 17.F)
      .in_high(file_content.size());

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_transfer>()  //
      |
      utl::for_each([&](csv_transfer const& t) {
        auto const from_stop_it = stops.find(t.from_stop_id_->view());
        if (from_stop_it == end(stops)) {
          log(log_lvl::error, "loader.gtfs.transfers", "stop {} not found\n",
              t.from_stop_id_->view());
          return;
        }

        auto const to_stop_it = stops.find(t.to_stop_id_->view());
        if (to_stop_it == end(stops)) {
          log(log_lvl::error, "loader.gtfs.transfers", "stop {} not found\n",
              t.to_stop_id_->view());
          return;
        }

        auto const type = static_cast<transfer_type>(*t.transfer_type_);
        if (type == transfer_type::kNotPossible || from_stop_it == to_stop_it) {
          return;
        }

        auto& footpaths = from_stop_it->second->footpaths_;
        auto const it = std::find_if(
            begin(footpaths), end(footpaths), [&](footpath const& fp) {
              return fp.target() == to_stop_it->second->location_;
            });
        if (it == end(footpaths)) {
          footpaths.emplace_back(
              footpath{to_stop_it->second->location_,
                       duration_t{*t.min_transfer_time_ / 60}});
        }
      });
}

locations_map read_stops(source_idx_t const src,
                         timetable& tt,
                         tz_map& timezones,
                         std::string_view stops_file_content,
                         std::string_view transfers_file_content,
                         unsigned link_stop_distance) {
  auto const timer = scoped_timer{"gtfs.loader.stops"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Stops")
      .out_bounds(1.F, 5.F)
      .in_high(stops_file_content.size());

  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_timezone")> timezone_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<utl::cstr, UTL_NAME("platform_code")> platform_code_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lat")> lat_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lon")> lon_;
  };

  locations_map locations;
  stop_map_t stops;
  hash_map<std::string_view, std::vector<stop*>> equal_names;
  utl::line_range{utl::make_buf_reader(stops_file_content,
                                       progress_tracker->update_fn())}  //
      | utl::csv<csv_stop>()  //
      |
      utl::for_each([&](csv_stop const& s) {
        auto const new_stop = utl::get_or_create(stops, s.id_->view(), [&]() {
                                return std::make_unique<stop>();
                              }).get();

        new_stop->id_ = s.id_->view();
        new_stop->name_ = s.name_->view();
        new_stop->coord_ = {utl::parse<double>(s.lat_->trim()),
                            utl::parse<double>(s.lon_->trim())};
        new_stop->platform_code_ = s.platform_code_->view();
        new_stop->timezone_ = s.timezone_->trim().view();

        if (!s.parent_station_->trim().empty()) {
          auto const parent =
              utl::get_or_create(stops, s.parent_station_->trim().view(), []() {
                return std::make_unique<stop>();
              }).get();
          parent->id_ = s.parent_station_->trim().to_str();
          parent->children_.emplace(new_stop);
          new_stop->parent_ = parent;
        }

        equal_names[s.name_->view()].emplace_back(new_stop);
      });

  auto const stop_vec =
      utl::to_vec(stops, [](auto const& s) { return s.second.get(); });
  for (auto const& [id, s] : stops) {
    for (auto const& equal : equal_names[s->name_]) {
      if (equal != s.get()) {
        s->same_name_.emplace(equal);
      }
    }
  }

  if (link_stop_distance != 0U) {
    auto const t = scoped_timer{"loader.gtfs.stop.rtree"};
    progress_tracker->status("Stops R-Tree")
        .out_bounds(5.F, 15.F)
        .in_high(stops.size());
    auto const stop_rtree = geo::make_point_rtree(
        stops, [](auto const& s) { return s.second->coord_; });
    utl::parallel_for(
        stops,
        [&](auto const& s) {
          s.second->compute_close_stations(stop_rtree, link_stop_distance);
        },
        progress_tracker->update_fn());
  }

  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};
  for (auto const& [id, s] : stops) {
    auto const is_track = s->parent_ != nullptr && !s->platform_code_.empty();
    locations.emplace(
        std::string{id},
        s->location_ = tt.locations_.register_location(location{
            id, is_track ? s->platform_code_ : s->name_, s->coord_, src,
            is_track ? location_type::kTrack : location_type::kStation,
            osm_node_id_t::invalid(), location_idx_t::invalid(),
            s->timezone_.empty() ? timezone_idx_t::invalid()
                                 : get_tz_idx(tt, timezones, s->timezone_),
            2_minutes, it_range{empty_idx_vec}, std::span{empty_footpath_vec},
            std::span{empty_footpath_vec}}));
  }

  read_transfers(stops, transfers_file_content);

  {
    auto const t = scoped_timer{"loader.gtfs.stop.metas"};
    progress_tracker->status("Compute Metas")
        .out_bounds(17.F, 20.F)
        .in_high(stops.size());

    auto const add_if_not_exists = [](auto bucket, footpath&& fp) {
      auto const it = std::find_if(begin(bucket), end(bucket), [&](auto&& x) {
        return fp.target() == x.target_;
      });
      if (it == end(bucket)) {
        bucket.emplace_back(fp);
      }
    };

    for (auto const& [id, s] : stops) {
      if (s->parent_ != nullptr) {
        tt.locations_.parents_[s->location_] = s->parent_->location_;
      }
      for (auto const& c : s->children_) {
        tt.locations_.children_[s->location_].emplace_back(c->location_);
      }

      // GTFS footpaths
      for (auto const& fp : s->footpaths_) {
        tt.locations_.preprocessing_footpaths_out_[s->location_].emplace_back(
            fp);
        tt.locations_.preprocessing_footpaths_in_[fp.target()].emplace_back(
            s->location_, fp.duration());
      }
    }

    // Make GTFS footpaths symmetric (if not already).
    for (auto const& [id, s] : stops) {
      for (auto const& fp : s->footpaths_) {
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_out_[fp.target()],
            {s->location_, fp.duration()});
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_in_[s->location_],
            {fp.target(), fp.duration()});
      }
    }

    // Generate footpaths to connect stops in close proximity.
    hash_set<stop*> todo, done;
    for (auto const& [id, s] : stops) {
      for (auto const& eq : s->get_metas(stop_vec, todo, done)) {
        tt.locations_.equivalences_[s->location_].emplace_back(eq->location_);
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_out_[s->location_],
            {eq->location_, 2_minutes});
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_in_[eq->location_],
            {s->location_, 2_minutes});
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_out_[eq->location_],
            {s->location_, 2_minutes});
        add_if_not_exists(
            tt.locations_.preprocessing_footpaths_in_[s->location_],
            {eq->location_, 2_minutes});
      }
      progress_tracker->increment();
    }
  }

  return locations;
}

}  // namespace nigiri::loader::gtfs

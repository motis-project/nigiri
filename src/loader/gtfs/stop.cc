#include "nigiri/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/loader/register.h"
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

    auto const lng_dist = geo::approx_distance_lng_degrees(coord_);

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
      auto const distance_in_m =
          geo::approx_squared_distance(meta->coord_, coord_, lng_dist);
      if ((distance_in_m > std::pow(500, 2) && !is_parent && !is_child) ||
          distance_in_m > std::pow(2000, 2)) {
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
  cista::raw::generic_string name_;
  std::string_view platform_code_;
  std::string_view stop_code_;
  std::string_view desc_;
  geo::latlng coord_;
  std::string_view timezone_;
  hash_set<stop*> same_name_, children_;
  stop* parent_{nullptr};
  std::vector<unsigned> close_;
  location_idx_t location_{location_idx_t::invalid()};
  bool wheelchair_boarding_;
};

enum class stop_type { kRegular, kGeneratedParent };
using stop_map_t = hash_map<std::string_view, std::unique_ptr<stop>>;

std::pair<stops_map_t, location_accessible_map_t> read_stops(
    source_idx_t const src,
    timetable& tt,
    translator& i18n,
    tz_map& timezones,
    std::string_view stops_file_content,
    unsigned link_stop_distance,
    duration_t const default_transfer_time,
    script_runner const& r) {
  auto const timer = scoped_timer{"gtfs.loader.stops"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Stops")
      .out_bounds(1.F, 5.F)
      .in_high(stops_file_content.size());

  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> id_;
    utl::csv_col<cista::raw::generic_string, UTL_NAME("stop_name")> name_;
    utl::csv_col<unsigned, UTL_NAME("location_type")> location_type_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_timezone")> timezone_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<utl::cstr, UTL_NAME("platform_code")> platform_code_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_code")> stop_code_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_desc")> stop_desc_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lat")> lat_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lon")> lon_;
    utl::csv_col<uint8_t, UTL_NAME("wheelchair_boarding")> wheelchair_boarding_;
  };

  stops_map_t locations;
  stop_map_t stops;
  hash_map<std::string_view, std::vector<stop*>> equal_names;
  utl::line_range{utl::make_buf_reader(stops_file_content,
                                       progress_tracker->update_fn())}  //
      | utl::csv<csv_stop>()  //
      |
      utl::for_each([&](csv_stop& s) {
        if (*s.location_type_ == 2 ||  // entrance / exit
            *s.location_type_ == 3  // generic node
        ) {
          return;
        }

        auto const new_stop = utl::get_or_create(stops, s.id_->view(), [&]() {
                                return std::make_unique<stop>();
                              }).get();

        new_stop->id_ = s.id_->view();
        new_stop->name_ = std::move(*s.name_);
        new_stop->coord_ = {
            std::clamp(utl::parse<double>(s.lat_->trim()), -90.0, 90.0),
            std::clamp(utl::parse<double>(s.lon_->trim()), -180.0, 180.0)};
        new_stop->platform_code_ = s.platform_code_->view();
        new_stop->stop_code_ = s.stop_code_->view();
        new_stop->desc_ = s.stop_desc_->view();
        new_stop->timezone_ = s.timezone_->trim().view();

        // we treat unknown as yes, because this data is currently not available
        // commonly enough
        new_stop->wheelchair_boarding_ = s.wheelchair_boarding_.val() != 2;

        if (!s.parent_station_->trim().empty()) {
          auto const parent =
              utl::get_or_create(stops, s.parent_station_->trim().view(), []() {
                return std::make_unique<stop>();
              }).get();
          parent->id_ = s.parent_station_->trim().view();
          parent->children_.emplace(new_stop);
          new_stop->parent_ = parent;
        }

        if (!new_stop->name_.empty()) {
          equal_names[new_stop->name_.view()].emplace_back(new_stop);
        }
      });

  auto const stop_vec =
      utl::to_vec(stops, [](auto const& s) { return s.second.get(); });
  for (auto const& [id, s] : stops) {
    if (s->name_.empty()) {
      continue;
    }
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

  auto accessible = location_accessible_map_t{};
  for (auto const& [id, s] : stops) {
    auto loc = location{
        tt,
        src,
        id,
        i18n.get(t::kStops, f::kStopName, s->name_.view(), s->id_),
        i18n.get(t::kStops, f::kPlatformCode, s->platform_code_, s->id_),
        i18n.get(t::kStops, f::kStopCode, s->stop_code_, s->id_),
        i18n.get(t::kStops, f::kStopDesc, s->desc_, s->id_),
        s->coord_,
        s->parent_ == nullptr ? location_type::kStation : location_type::kTrack,
        location_idx_t::invalid(),
        s->timezone_.empty() ? timezone_idx_t::invalid()
                             : get_tz_idx(tt, timezones, s->timezone_),
        default_transfer_time,
        timezones};
    if (process_location(r, loc)) {
      locations.emplace(id, s->location_ = register_location(tt, loc));
      accessible.insert({s->location_, s->wheelchair_boarding_});
    }
  }

  {
    auto const t = scoped_timer{"loader.gtfs.stop.metas"};
    progress_tracker->status("Compute Metas")
        .out_bounds(17.F, 20.F)
        .in_high(stops.size());

    for (auto const& [id, s] : stops) {
      if (s->parent_ != nullptr) {
        tt.locations_.parents_[s->location_] = s->parent_->location_;
      }
      for (auto const& c : s->children_) {
        tt.locations_.children_[s->location_].emplace_back(c->location_);
      }
    }

    // Collect equivalent stops in close proximity (used for start/destination
    // expansion; walking transfers between them are derived in
    // loader::build_footpaths).
    hash_set<stop*> todo, done;
    for (auto const& [id, s] : stops) {
      for (auto const& eq : s->get_metas(stop_vec, todo, done)) {
        tt.locations_.equivalences_[s->location_].emplace_back(eq->location_);
      }
      progress_tracker->increment();
    }
  }

  return std::pair{std::move(locations), std::move(accessible)};
}

}  // namespace nigiri::loader::gtfs

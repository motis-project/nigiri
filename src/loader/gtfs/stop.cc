#include "nigiri/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "geo/point_rtree.h"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/to_vec.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

struct stop {
  void compute_close_stations(geo::point_rtree const& stop_rtree) {
    if (std::abs(coord_.lat_) < 2.0 && std::abs(coord_.lng_) < 2.0) {
      return;
    }
    close_ = utl::to_vec(
        stop_rtree.in_radius(coord_, 100),
        [](std::size_t const idx) { return static_cast<unsigned>(idx); });
  }

  std::set<stop*> get_metas(std::vector<stop*> const& stops) {
    std::set<stop*> todo, done;
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
  location_idx_t location_;
  std::vector<footpath> footpaths_;
};

enum class transfer_type : std::uint8_t {
  kRecommended = 0U,
  kTimed = 1U,
  kMinimumChangeTime = 2U,
  kNotPossible = 3U,
  kStaySeated = 4U,
  kNoStaySeated = 5U,
  kGenerated = std::numeric_limits<std::uint8_t>::max()
};

void read_transfers(hash_map<std::string_view, std::unique_ptr<stop>>& stops,
                    std::string_view file_content) {
  nigiri::scoped_timer timer{"read transfers"};

  struct csv_transfer {
    utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_stop_id_;
    utl::csv_col<int, UTL_NAME("transfer_type")> transfer_type_;
    utl::csv_col<int, UTL_NAME("min_transfer_time")> min_transfer_time_;
  };

  if (file_content.empty()) {
    return;
  }

  utl::line_range{utl::buf_reader{file_content}}  //
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

        from_stop_it->second->footpaths_.emplace_back(
            footpath{.target_ = to_stop_it->second->location_,
                     .duration_ = duration_t{*t.min_transfer_time_ / 60}});
      });
}

locations_map read_stops(source_idx_t const src,
                         timetable& tt,
                         tz_map& timezones,
                         std::string_view stops_file_content,
                         std::string_view transfers_file_content) {
  scoped_timer timer{"read stops"};

  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_timezone")> timezone_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<utl::cstr, UTL_NAME("platform_code")> platform_code_;
    utl::csv_col<float, UTL_NAME("stop_lat")> lat_;
    utl::csv_col<float, UTL_NAME("stop_lon")> lon_;
  };

  locations_map locations;
  hash_map<std::string_view, std::unique_ptr<stop>> stops;
  hash_map<std::string_view, std::vector<stop*>> equal_names;
  utl::line_range{utl::buf_reader{stops_file_content}}  //
      | utl::csv<csv_stop>()  //
      |
      utl::for_each([&](csv_stop const& s) {
        auto const new_stop = utl::get_or_create(stops, s.id_->view(), [&]() {
                                return std::make_unique<stop>();
                              }).get();

        new_stop->id_ = s.id_->view();
        new_stop->name_ = s.name_->view();
        new_stop->coord_ = {*s.lat_, *s.lon_};
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

  auto const stop_rtree = geo::make_point_rtree(
      stops, [](auto const& s) { return s.second->coord_; });
  utl::parallel_for(stops, [&](auto const& s) {
    s.second->compute_close_stations(stop_rtree);
  });

  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};
  for (auto const& [id, s] : stops) {
    auto const is_track = s->parent_ != nullptr && !s->platform_code_.empty();
    locations.emplace(
        std::string{id},
        tt.locations_.register_location(location{
            is_track ? s->platform_code_ : s->id_, s->name_, s->coord_, src,
            is_track ? location_type::kTrack : location_type::kStation,
            osm_node_id_t::invalid(), location_idx_t::invalid(),
            get_tz_idx(tt, timezones, s->timezone_), 2_minutes,
            it_range{empty_idx_vec}, it_range{empty_footpath_vec},
            it_range{empty_footpath_vec}}));
  }

  read_transfers(stops, transfers_file_content);

  for (auto const& [id, s] : stops) {
    if (s->parent_ != nullptr) {
      tt.locations_.parents_[s->location_] = s->parent_->location_;
    }
    for (auto const& c : s->children_) {
      tt.locations_.children_[s->location_].emplace_back(c->location_);
    }
    for (auto const& eq : s->get_metas(stop_vec)) {
      tt.locations_.equivalences_[s->location_].emplace_back(eq->location_);
      tt.locations_.footpaths_out_[s->location_].emplace_back(eq->location_,
                                                              2_minutes);
      tt.locations_.footpaths_in_[eq->location_].emplace_back(s->location_,
                                                              2_minutes);
    }
  }

  return locations;
}

}  // namespace nigiri::loader::gtfs

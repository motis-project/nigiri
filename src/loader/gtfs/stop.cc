#include "nigiri/loader/gtfs/stop.h"

#include <algorithm>
#include <string>
#include <tuple>

#include "utl/get_or_create.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/to_vec.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

void stop::compute_close_stations(geo::point_rtree const& stop_rtree) {
  if (std::abs(coord_.lat_) < 2.0 && std::abs(coord_.lng_) < 2.0) {
    return;
  }
  close_ = utl::to_vec(
      stop_rtree.in_radius(coord_, 100),
      [](std::size_t const idx) { return static_cast<unsigned>(idx); });
}

std::set<stop*> stop::get_metas(std::vector<stop*> const& stops) {
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

    for (auto const& p : next->parents_) {
      if (done.find(p) == end(done)) {
        todo.emplace(p);
      }
    }

    for (auto const& p : next->children_) {
      if (done.find(p) == end(done)) {
        todo.emplace(p);
      }
    }
  }

  for (auto it = begin(done); it != end(done);) {
    auto* meta = *it;
    auto const is_parent = parents_.find(meta) != end(parents_);
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

stop_map read_stops(std::string_view file_content) {
  scoped_timer timer{"read stops"};

  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_timezone")> timezone_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<float, UTL_NAME("stop_lat")> lat_;
    utl::csv_col<float, UTL_NAME("stop_lon")> lon_;
  };

  stop_map stops;
  hash_map<std::string, std::vector<stop*>> equal_names;
  utl::line_range{utl::buf_reader{file_content}}  //
      | utl::csv<csv_stop>()  //
      |
      utl::for_each([&](csv_stop const& s) {
        auto const new_stop = utl::get_or_create(stops, s.id_->view(), [&]() {
                                return std::make_unique<stop>();
                              }).get();

        new_stop->id_ = s.id_->to_str();
        new_stop->name_ = s.name_->to_str();
        new_stop->coord_ = {*s.lat_, *s.lon_};
        new_stop->timezone_ = s.timezone_->to_str();

        if (!s.parent_station_->trim().empty()) {
          auto const parent =
              utl::get_or_create(stops, s.parent_station_->trim().view(), []() {
                return std::make_unique<stop>();
              }).get();
          parent->id_ = s.parent_station_->trim().to_str();
          parent->children_.emplace(new_stop);
          new_stop->parents_.emplace(parent);
        }

        equal_names[s.name_->view()].emplace_back(new_stop);
      });

  for (auto const& [id, s] : stops) {
    for (auto const& equal : equal_names[s->name_]) {
      if (equal != s.get()) {
        s->same_name_.emplace(equal);
      }
    }
  }

  return stops;
}

}  // namespace nigiri::loader::gtfs

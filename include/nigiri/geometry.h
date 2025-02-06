#pragma once

#include <cista/containers/vector.h>
#include <geo/box.h>
#include <tg.h>
#include <tuple>

namespace nigiri {
template <typename T>
using vector = cista::raw::vector<T>;

struct point {
  point() {}

  point(double const x, double const y) : x_{x}, y_{y} {}

  bool intersects(geo::box const& b) const;

  auto cista_members() { return std::tie(x_, y_); }
  double x_;
  double y_;
};

struct ring {
  ring() {}

  ring(point& p) { points_.emplace_back(p); }

  ring(point&& p) { points_.emplace_back(p); }

  ring(std::initializer_list<point> points) {
    points_.reserve(points.size());
    for (auto& p : points) {
      points_.emplace_back(p);
    }
  }

  void as_points(std::vector<point*>& points) {
    points.reserve(points.size() + points_.size());
    for (auto& p : points_) {
      points.emplace_back(&p);
    }
  }

  bool intersects(geo::box const& b) const;

  auto cista_members() { return std::tie(points_); }
  vector<point> points_;
};

struct polygon {
  polygon() {}

  polygon(ring& exterior) : exterior_(exterior), holes_(vector<ring>{}) {}

  polygon(ring&& exterior) : exterior_(exterior), holes_(vector<ring>{}) {}

  polygon(ring&& exterior, std::initializer_list<ring> holes)
      : exterior_(exterior) {
    holes_.reserve(holes.size());
    for (auto& h : holes) {
      holes_.emplace_back(h);
    }
  }

  bool intersects(geo::box const& b) const;

  void as_points(std::vector<point*>& points) { exterior_.as_points(points); }

  auto cista_members() { return std::tie(exterior_, holes_); }
  ring exterior_;
  vector<ring> holes_;
};

struct multipolgyon {
  multipolgyon() : original_type_(TG_MULTIPOLYGON) {}

  multipolgyon(polygon& polygon, tg_geom_type original_type)
      : original_type_(original_type) {
    polygons_.push_back(polygon);
  }

  multipolgyon(polygon&& polygon, tg_geom_type originaltype)
      : original_type_(originaltype) {
    polygons_.push_back(polygon);
  }

  multipolgyon(std::initializer_list<polygon> polygons) {
    polygons_.reserve(polygons.size());
    for (auto& p : polygons) {
      polygons_.emplace_back(p);
    }
  }

  void as_points(std::vector<point*>& points) {
    for (auto& p : polygons_) {
      p.as_points(points);
    }
  }

  geo::latlng get_center();

  tg_geom* to_tg_geom();

  bool intersects(geo::box const& b) const;

  geo::box bounding_box();

  auto cista_members() { return std::tie(polygons_, original_type_); }
  vector<polygon> polygons_;
  tg_geom_type original_type_;
};

ring create_ring(tg_ring const* ring);

polygon create_polygon(tg_poly const* poly);

multipolgyon create_multipolygon(tg_geom const* multi);

tg_point create_tg_point(point const& point);

tg_ring* create_tg_ring(ring const& ring);

tg_poly* create_tg_poly(polygon const& poly);

tg_geom* create_tg_multipoly(multipolgyon const& multipoly);

multipolgyon point_to_multipolygon(point& point);

multipolgyon point_to_multipolygon(point&& point);

multipolgyon ring_to_multipolygon(ring& ring);

multipolgyon ring_to_multipolygon(ring&& ring);

multipolgyon polygon_to_multipolygon(polygon& polygon);

multipolgyon polygon_to_multipolygon(polygon&& polygon);

point point_from_multipolygon(multipolgyon& multipoly);

ring ring_from_multipolygon(multipolgyon& multipoly);

polygon polygon_from_multipolygon(multipolgyon& multipoly);

};  // namespace nigiri
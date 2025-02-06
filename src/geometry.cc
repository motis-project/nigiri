#include "nigiri/geometry.h"

#include <nigiri/logging.h>

namespace nigiri {

ring create_ring(tg_ring const* ring) {
  auto const* points = tg_ring_points(ring);
  auto const num_points = tg_ring_num_points(ring);

  nigiri::ring r{};
  r.points_.reserve(num_points);
  for (auto i = 0; i < num_points; ++i, ++points) {
    r.points_.emplace_back(points->x, points->y);
  }
  return r;
};

polygon create_polygon(tg_poly const* poly) {
  auto const* exterior = tg_poly_exterior(poly);
  auto const num_holes = tg_poly_num_holes(poly);

  polygon p{create_ring(exterior)};
  for (auto i = 0; i < num_holes; ++i) {
    auto const hole = tg_poly_hole_at(poly, i);
    p.holes_.push_back(create_ring(hole));
  }
  return p;
};

multipolgyon create_multipolygon(tg_geom const* multi) {
  auto const num_polys = tg_geom_num_polys(multi);
  nigiri::multipolgyon m{};
  for (auto i = 0; i < num_polys; ++i) {
    m.polygons_.push_back(create_polygon(tg_geom_poly_at(multi, i)));
  }
  return m;
};

tg_point create_tg_point(point const& point) {
  return tg_point(point.x_, point.y_);
}

tg_ring* create_tg_ring(ring const& ring) {
  if (ring.points_.empty()) {
    return nullptr;
  }
  std::vector<tg_point> points;
  points.reserve(ring.points_.size());
  for (auto i = 0; i < ring.points_.size(); ++i) {
    points.emplace_back(ring.points_.at(i).x_, ring.points_.at(i).y_);
  }
  auto* pointer = &points[0];
  return tg_ring_new(pointer, ring.points_.size());
}

tg_poly* create_tg_poly(polygon const& poly) {
  tg_ring* exterior = create_tg_ring(poly.exterior_);
  if (exterior == nullptr) {
    return nullptr;
  }
  std::vector<tg_ring*> rings;
  rings.reserve(poly.holes_.size());
  if (poly.holes_.empty()) {
    return tg_poly_new(exterior, nullptr, 0);
  }
  for (auto i = 0; i < poly.holes_.size(); ++i) {
    rings.push_back(create_tg_ring(poly.holes_.at(i)));
  }
  auto* pointer = &rings[0];
  return tg_poly_new(exterior, pointer, poly.holes_.size());
}

tg_geom* create_tg_multipoly(multipolgyon const& multipoly) {
  int const num_poly = multipoly.polygons_.size();
  std::vector<tg_poly*> polygons;
  polygons.reserve(num_poly);
  for (auto i = 0; i < num_poly; ++i) {
    auto* poly = create_tg_poly(multipoly.polygons_[i]);
    polygons.emplace_back(poly);
  }
  return tg_geom_new_multipolygon(polygons.data(), num_poly);
}

multipolgyon point_to_multipolygon(point& point) {
  return multipolgyon{polygon{ring{point}}, TG_POINT};
}

multipolgyon point_to_multipolygon(point&& point) {
  return point_to_multipolygon(point);
}

multipolgyon ring_to_multipolygon(ring& ring) {
  return multipolgyon{polygon{ring}, TG_MULTIPOINT};
}

multipolgyon ring_to_multipolygon(ring&& ring) {
  return ring_to_multipolygon(ring);
}

multipolgyon polygon_to_multipolygon(polygon& polygon) {
  return multipolgyon{polygon, TG_POLYGON};
}

multipolgyon polygon_to_multipolygon(polygon&& polygon) {
  return polygon_to_multipolygon(polygon);
}

point point_from_multipolygon(multipolgyon& multipoly) {
  if (multipoly.polygons_.empty()) {
    log(log_lvl::error, "geometry.point_from_multipoly", "Empty Multipolygon");
    return point{0, 0};
  }
  if (multipoly.polygons_[0].exterior_.points_.empty()) {
    log(log_lvl::error, "geometry.point_from_multipoly", "Empty Ring");
    return point{0, 0};
  }
  return multipoly.polygons_[0].exterior_.points_[0];
}

ring ring_from_multipolygon(multipolgyon& multipoly) {
  if (multipoly.polygons_.empty()) {
    log(log_lvl::error, "geometry.point_from_multipoly", "Empty Multipolygon");
    return ring{};
  }
  return multipoly.polygons_[0].exterior_;
}

polygon polygon_from_multipolygon(multipolgyon& multipoly) {
  if (multipoly.polygons_.empty()) {
    log(log_lvl::error, "geometry.point_from_multipoly", "Empty Multipolygon");
    return polygon{};
  }
  return multipoly.polygons_[0];
}

tg_geom* multipolgyon::to_tg_geom() {
  switch (original_type_) {
    case TG_POINT:
      return tg_geom_new_point(create_tg_point(point_from_multipolygon(*this)));
    case TG_MULTIPOINT:
      return reinterpret_cast<tg_geom*>(
          create_tg_ring(ring_from_multipolygon(*this)));
    case TG_POLYGON:
      return reinterpret_cast<tg_geom*>(
          create_tg_poly(polygon_from_multipolygon(*this)));
    case TG_MULTIPOLYGON: return create_tg_multipoly(*this);
    default: {
      log(log_lvl::error, "loader.geometry.to_tg_geom", "Unknown type {}",
          static_cast<int>(original_type_));
      return nullptr;
    }
  }
}

bool point::intersects(geo::box const& b) const {
  return b.contains(geo::latlng{this->x_, this->y_});
}

bool ring::intersects(geo::box const& b) const {
  auto const tg_box = tg_rect{.min = tg_point{b.min_.lng(), b.min_.lat()},
                              .max = tg_point{b.max_.lng(), b.max_.lat()}};
  auto const r = reinterpret_cast<tg_geom*>(create_tg_ring(*this));
  bool const result = tg_geom_intersects_rect(r, tg_box);
  tg_geom_free(r);
  return result;
}

bool polygon::intersects(geo::box const& b) const {
  auto const tg_box = tg_rect{.min = tg_point{b.min_.lng(), b.min_.lat()},
                              .max = tg_point{b.max_.lng(), b.max_.lat()}};
  auto const p = reinterpret_cast<tg_geom*>(create_tg_poly(*this));
  bool const result = tg_geom_intersects_rect(p, tg_box);
  tg_geom_free(p);
  return result;
}

bool multipolgyon::intersects(geo::box const& b) const {
  auto const tg_box = tg_rect{.min = tg_point{b.min_.lng(), b.min_.lat()},
                              .max = tg_point{b.max_.lng(), b.max_.lat()}};
  auto const m = create_tg_multipoly(*this);
  bool const result = tg_geom_intersects_rect(m, tg_box);
  tg_geom_free(m);
  return result;
}

geo::box multipolgyon::bounding_box() {
  std::vector<point*> points{};
  as_points(points);
  auto box = geo::box{};
  for (auto const p : points) {
    box.extend(geo::latlng{p->y_, p->x_});
  }
  return box;
}

geo::latlng multipolgyon::get_center() {
  auto const tg = to_tg_geom();
  auto [x, y] = tg_geom_point(tg);
  tg_geom_free(tg);
  return geo::latlng{.lat_ = y, .lng_ = x};
}

};  // namespace nigiri
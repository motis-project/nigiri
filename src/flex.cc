#include "nigiri/flex.h"

#include "geo/detail/register_latlng.h"

#include "boost/geometry/algorithms/within.hpp"
#include "boost/geometry/core/cs.hpp"
#include "boost/geometry/geometries/register/multi_polygon.hpp"
#include "boost/geometry/geometries/register/point.hpp"
#include "boost/geometry/geometries/register/ring.hpp"

#include "nigiri/timetable.h"

namespace nigiri {

using inner_rings_t = nvec<area_idx_t, geo::latlng, 3U>;
using outer_rings_t = nvec<area_idx_t, geo::latlng, 2U>;
using ring_t = cista::const_bucket<inner_rings_t::data_vec_t,
                                   inner_rings_t::index_vec_t,
                                   inner_rings_t::size_type>;

struct polygon {
  using point_type = geo::latlng;
  using ring_type = ring_t;
  using inner_container_type =
      cista::const_meta_bucket<1U,
                               inner_rings_t::data_vec_t,
                               inner_rings_t::index_vec_t,
                               inner_rings_t::size_type>;

  ring_type outer() const { return outer_; }
  inner_container_type inners() const { return inners_; }

  ring_type outer_;
  inner_container_type inners_;
};

using multi_polygon = std::vector<polygon>;

}  // namespace nigiri

BOOST_GEOMETRY_REGISTER_RING(nigiri::ring_t)
BOOST_GEOMETRY_REGISTER_MULTI_POLYGON(nigiri::multi_polygon)

namespace boost::geometry::traits {

template <>
struct tag<nigiri::polygon> {
  using type = polygon_tag;
};

template <>
struct ring_const_type<nigiri::polygon> {
  using type = nigiri::ring_t;
};

template <>
struct ring_mutable_type<nigiri::polygon> {
  using type = nigiri::ring_t;
};

template <>
struct interior_const_type<nigiri::polygon> {
  using type = nigiri::polygon::inner_container_type;
};

template <>
struct interior_mutable_type<nigiri::polygon> {
  using type = nigiri::polygon::inner_container_type;
};

template <>
struct exterior_ring<nigiri::polygon> {
  static nigiri::polygon::ring_type get(nigiri::polygon const& p) {
    return p.outer();
  }
};

template <>
struct interior_rings<nigiri::polygon> {
  static nigiri::polygon::inner_container_type get(nigiri::polygon const& p) {
    return p.inners();
  }
};

template <>
struct point_order<nigiri::ring_t> {
  static order_selector const value = counterclockwise;
};

template <>
struct closure<nigiri::ring_t> {
  static closure_selector const value = closed;
};

}  // namespace boost::geometry::traits

namespace nigiri {

multi_polygon get_area(timetable const& tt, flex_area_idx_t const area) {
  utl::verify(
      tt.flex_area_outers_[area].size() == tt.flex_area_inners_[area].size(),
      "size mismatch: n_outers={}, n_inners={}",
      tt.flex_area_outers_[area].size(), tt.flex_area_inners_[area].size());
  auto const size = tt.flex_area_outers_[area].size();
  auto mp = multi_polygon{};
  for (auto i = 0U; i != size; ++i) {
    mp.emplace_back(tt.flex_area_outers_[area][i],
                    tt.flex_area_inners_[area][i]);
  }
  return mp;
}

bool is_in_flex_area(timetable const& tt,
                     flex_area_idx_t const a,
                     geo::latlng const& pos) {
  return tt.flex_area_bbox_[a].contains(pos) &&
         boost::geometry::within(pos, get_area(tt, a));
}

}  // namespace nigiri
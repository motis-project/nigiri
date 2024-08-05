#pragma once

#include <cstddef>
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"
#include "cista/mmap.h"
#include "geo/latlng.h"

// #include "osmium/osm/location.hpp"
// #include "osr/types.h"

namespace nigiri::loader::gtfs {
namespace helper {

/* Code duplicated from 'osr/types.h' */
template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename K, typename V, typename SizeType = cista::base_t<K>>
using mm_vecvec = cista::basic_vecvec<K, mm_vec<V>, mm_vec<SizeType>>;

/* Code duplicated from 'osmium/osm/location.hpp' */
constexpr int32_t coordinate_precision{10000000};

constexpr int32_t double_to_fix(double const c) noexcept {
  return static_cast<int32_t>(std::round(c * coordinate_precision));
}

constexpr double fix_to_double(int32_t const c) noexcept {
  return static_cast<double>(c) / coordinate_precision;
}
}  // namespace helper

class ShapeMap {
public:
  using key_type = std::string_view;
  using value_type = std::vector<geo::latlng>;
  struct Paths;
  struct Iterator;

  ShapeMap(std::string_view const, Paths const&);
  size_t size() const;
  bool contains(key_type const&) const;
  Iterator begin() const;
  Iterator end() const;
  value_type at(key_type const&) const;
  static void write_shapes(std::string_view const, Paths const&);

private:
  using shape_coordinate_type =
      std::remove_const<decltype(helper::coordinate_precision)>::type;
  struct Coordinate {
    shape_coordinate_type lat, lon;
    bool operator==(Coordinate const& other) const = default;
  };
  using shape_data_t = helper::mm_vecvec<std::size_t, ShapeMap::Coordinate>;
  using id_vec_t = std::vector<key_type>;
  using id_map_t = std::unordered_map<key_type, size_t>;

  ShapeMap(std::pair<shape_data_t, id_vec_t>);
  static std::pair<shape_data_t, id_vec_t> create_files(std::string_view const,
                                                        Paths const&);
  static shape_data_t create_memory_map(
      Paths const&,
      cista::mmap::protection const = cista::mmap::protection::READ);
  static auto create_id_memory_map(
      std::filesystem::path const&,
      cista::mmap::protection const = cista::mmap::protection::READ);
  static id_vec_t load_shapes(std::string_view const, shape_data_t&);
  static id_map_t id_vec_to_map(id_vec_t const&);
  static value_type transform_coordinates(auto const&);

  shape_data_t const shape_map_;
  id_map_t const id_map_;

  friend struct ShapePoint;
};

struct ShapeMap::Paths {
  std::filesystem::path id_file;
  std::filesystem::path shape_data_file;
  std::filesystem::path shape_metadata_file;
};

struct ShapeMap::Iterator {
  using difference_type = std::ptrdiff_t;
  using value_type = std::pair<ShapeMap::key_type, ShapeMap::value_type>;
  bool operator==(Iterator const&) const = default;
  Iterator& operator++();
  Iterator operator++(int);
  value_type operator*() const;
  ShapeMap const* shapes;
  ShapeMap::id_map_t::const_iterator it;
};
static_assert(std::forward_iterator<ShapeMap::Iterator>);

}  // namespace nigiri::loader::gtfs
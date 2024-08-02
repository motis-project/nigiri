#include "nigiri/loader/gtfs/shape.h"
#include <sys/types.h>

#include <format>
#include <ranges>

#include "cista/mmap.h"
#include "geo/latlng.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

namespace nigiri::loader::gtfs {

InvalidShapesFormat::InvalidShapesFormat(std::string const& msg)
    : std::runtime_error{msg} {}

struct ShapePoint {
  ShapeMap::key_type const id;
  ShapeMap::Coordinate const coordinate;
  size_t const seq;
  struct Entry {
    utl::csv_col<ShapeMap::key_type, UTL_NAME("shape_id")> id;
    utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat;
    utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon;
    utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq;
  };
  static constexpr ShapePoint from_entry(Entry const&);
};

constexpr ShapePoint ShapePoint::from_entry(Entry const& entry) {
  return ShapePoint{
      entry.id.val(),
      {
          helper::double_to_fix(entry.lat.val()),
          helper::double_to_fix(entry.lon.val()),
      },
      entry.seq.val(),
  };
}

ShapeMap::ShapeMap(std::string_view const data, Paths const& paths)
    : ShapeMap(create_files(data, paths)) {}

ShapeMap::ShapeMap(Paths const& paths) : ShapeMap(load_files(paths)) {}

ShapeMap::ShapeMap(std::pair<shape_data_t, id_vec_t> p)
    : shape_map_{std::move(p.first)}, id_map_{id_vec_to_map(p.second)} {}

size_t ShapeMap::size() const { return id_map_.size(); }

bool ShapeMap::contains(key_type const& id) const {
  return id_map_.contains(id);
}

ShapeMap::Iterator ShapeMap::begin() const {
  return Iterator{
      this,
      id_map_.begin(),
  };
}

ShapeMap::Iterator ShapeMap::end() const {
  return Iterator{
      this,
      id_map_.end(),
  };
}

ShapeMap::value_type ShapeMap::at(key_type const& id) const {
  auto offset = id_map_.at(id);
  return transform_coordinates(shape_map_.at(offset));
}

void ShapeMap::write_shapes(std::string_view const data, Paths const& paths) {
  create_files(data, paths);
}

std::pair<ShapeMap::shape_data_t, ShapeMap::id_vec_t> ShapeMap::create_files(
    std::string_view const data, Paths const& paths) {
  shape_data_t mmap{create_memory_map(paths, cista::mmap::protection::WRITE)};
  id_vec_t ids{load_shapes(data, mmap)};
  store_ids(ids, paths.id_file);
  return std::make_pair(std::move(mmap), std::move(ids));
}

std::pair<ShapeMap::shape_data_t, ShapeMap::id_vec_t> ShapeMap::load_files(
    Paths const& paths) {
  shape_data_t mmap{create_memory_map(paths)};
  return std::make_pair(std::move(mmap), load_ids(paths.id_file));
}

ShapeMap::shape_data_t ShapeMap::create_memory_map(
    Paths const& paths, cista::mmap::protection const mode) {
  return shape_data_t{
      cista::basic_mmap_vec<Coordinate, std::size_t>{
          cista::mmap{paths.shape_data_file.native().data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{
          cista::mmap{paths.shape_metadata_file.native().data(), mode}}};
}

auto ShapeMap::create_id_memory_map(std::filesystem::path const& path,
                                    cista::mmap::protection const mode) {
  return cista::mmap_vec<key_type::value_type>{
      cista::mmap{path.native().data(), mode}};
}

ShapeMap::id_vec_t ShapeMap::load_shapes(std::string_view const data,
                                         shape_data_t& mmap) {
  struct BucketState {
    size_t offset{};
    size_t last_seq{};
  };
  std::unordered_map<key_type, BucketState> states;
  id_vec_t ids;

  auto store_to_map = [&ids, &mmap, &states](ShapePoint const point) {
    if (auto found = states.find(point.id); found != states.end()) {
      auto& state = found->second;
      if (state.last_seq >= point.seq) {
        throw InvalidShapesFormat(
            std::format("Non monotonic sequence for shape_id '{}': Sequence "
                        "number {} followed by {}",
                        point.id, state.last_seq, point.seq));
      }
      mmap[state.offset].push_back(point.coordinate);
      state.last_seq = point.seq;
    } else {
      auto offset = ids.size();
      auto bucket = mmap.add_back_sized(0u);
      states.insert({point.id, {offset, point.seq}});
      bucket.push_back(point.coordinate);
      ids.push_back(point.id);
    }
  };

  utl::line_range{utl::make_buf_reader(data, utl::noop_progress_consumer{})} |
      utl::csv<ShapePoint::Entry>() |
      utl::transform([&](ShapePoint::Entry const& entry) {
        return ShapePoint::from_entry(entry);
      }) |
      utl::for_each(store_to_map);

  return ids;
}
void ShapeMap::store_ids(id_vec_t const& ids,
                         std::filesystem::path const& path) {
  auto storage{create_id_memory_map(path, cista::mmap::protection::WRITE)};
  for (auto id : ids) {
    storage.insert(storage.end(), id.begin(), id.end());
    storage.push_back('\0');
  }
}

ShapeMap::id_vec_t ShapeMap::load_ids(std::filesystem::path const& path) {
  id_vec_t ids{};
  auto storage{create_id_memory_map(path)};
  std::string_view view{storage};
  size_t start{0u}, end;
  while ((end = view.find('\0', start)) != view.npos) {
    ids.push_back(key_type{view.substr(start, end - start)});
    start = end + 1;
  }
  return ids;
}

ShapeMap::id_map_t ShapeMap::id_vec_to_map(id_vec_t const& ids) {
  id_map_t map;
  for (auto [pos, id] : std::ranges::enumerate_view(ids)) {
    map.insert({id, pos});
  }
  return map;
}

ShapeMap::value_type ShapeMap::transform_coordinates(auto const& shape) {
  auto coordinates = shape | std::views::transform([](Coordinate const& c) {
                       return geo::latlng{helper::fix_to_double(c.lat),
                                          helper::fix_to_double(c.lon)};
                     });
  return value_type{coordinates.begin(), coordinates.end()};
}

ShapeMap::Iterator& ShapeMap::Iterator::operator++() {
  ++it;
  return *this;
}

ShapeMap::Iterator ShapeMap::Iterator::operator++(int) {
  auto tmp(*this);
  ++(*this);
  return tmp;
}

ShapeMap::value_type ShapeMap::Iterator::operator*() const {
  return shapes->at(it->first);
}
}  // namespace nigiri::loader::gtfs

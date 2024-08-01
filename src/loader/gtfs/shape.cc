#include "nigiri/loader/gtfs/shape.h"

#include <algorithm>
#include <ranges>
#include "geo/latlng.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

namespace nigiri::loader::gtfs {

    ShapeMap::ShapeMap(const std::string_view data, const Paths& paths) : ShapeMap(create_files(data, paths)) {}

    ShapeMap::ShapeMap(std::pair<shape_data_t, id_vec_t> p) : shape_map_{std::move(p.first)}, id_map_{id_vec_to_map(p.second)} {}

    size_t ShapeMap::size() const {
        return id_map_.size();
    }

    bool ShapeMap::contains(const key_type& id) const {
        return id_map_.contains(id);
    }

    std::vector<geo::latlng> ShapeMap::at(const key_type& id) const {
        auto offset = id_map_.at(id);
        return transform_coordinates(shape_map_.at(offset));
    }

    std::pair<ShapeMap::shape_data_t, ShapeMap::id_vec_t> ShapeMap::create_files(const std::string_view data, const Paths& paths) {
        shape_data_t mmap{create_memory_map(paths, cista::mmap::protection::WRITE)};
        id_vec_t ids{load_shapes(data, mmap)};
        store_ids(ids, paths.id_file);
        return std::make_pair(std::move(mmap), std::move(ids));
    }

    ShapeMap::shape_data_t ShapeMap::create_memory_map(const Paths& paths, const cista::mmap::protection mode = cista::mmap::protection::READ) {
        return shape_data_t{
            cista::basic_mmap_vec<Coordinate, std::size_t>{
                cista::mmap{paths.shape_data_file.native().data(), mode}},
            cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{
                paths.shape_metadata_file.native().data(), mode}}};
    }

    ShapeMap::id_vec_t ShapeMap::load_shapes(const std::string_view data, shape_data_t& mmap) {
        key_type last_id;
        auto bucket = mmap.add_back_sized(0u);
        id_vec_t ids;

        auto store_to_map = [&bucket, &mmap, &ids, &last_id](const auto& point) {
            if (last_id.empty()) {
                ids.push_back(point.id);
                last_id = point.id;
            } else if (last_id != point.id) {
                ids.push_back(point.id);
                bucket = mmap.add_back_sized(0u);
                last_id = point.id;
            }
            bucket.push_back(point.coordinate);
        };

        utl::line_range{utl::make_buf_reader(data, utl::noop_progress_consumer{})}
            | utl::csv<ShapePoint::Entry>()
            | utl::transform([&](ShapePoint::Entry const& entry) {
                return ShapePoint::from_entry(entry);
              })
            | utl::for_each(store_to_map);

        return ids;
    }
    void ShapeMap::store_ids(const id_vec_t&, const std::filesystem::path&) {
        // TODO
    }
    ShapeMap::id_map_t ShapeMap::id_vec_to_map(const id_vec_t& ids) {
        id_map_t map;
        for (auto [pos, id] : std::ranges::enumerate_view(ids)) {
            map.insert({id, pos});
        }
        return map;
    }

    std::vector<geo::latlng> ShapeMap::transform_coordinates(const auto& shape) {
        std::vector<geo::latlng> coordinates;
        for (auto point : shape) {
            coordinates.push_back(geo::latlng{helper::fix_to_double(point.lat), helper::fix_to_double(point.lon)});
        }
        return coordinates;
        // auto coordinates = shape
        //     | std::views::transform([](const Coordinate& c) {
        //         return geo::latlng{helper::fix_to_double(c.lat), helper::fix_to_double(c.lon)};
        //     });
        // return std::vector<geo::latlng>{coordinates};
    }

    constexpr ShapeMap::ShapePoint ShapeMap::ShapePoint::from_entry(const Entry& entry) {
        return ShapePoint{
            entry.id.val(),
            {
                helper::double_to_fix(entry.lat.val()),
                helper::double_to_fix(entry.lon.val()),
            },
            entry.seq.val(),
        };
    }

}

#pragma once

#include "nigiri/types.h"

#include "cista//containers/vector.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

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

        constexpr int32_t double_to_fix(const double c) noexcept {
            return static_cast<int32_t>(std::round(c * coordinate_precision));
        }

        constexpr double fix_to_double(const int32_t c) noexcept {
            return static_cast<double>(c) / coordinate_precision;
        }
    }


    using shape_coordinate_type = int32_t;
    using shape_id_type = std::string;
    using shape_id_vec = std::vector<shape_id_type>;


    struct ShapePoint {
        const shape_id_type id;
        const struct Coordinate {
            shape_coordinate_type lat, lon;
            bool operator==(const Coordinate& other) const = default;
        } coordinate;
        const size_t seq;

        struct Shape {
            utl::csv_col<shape_id_type, UTL_NAME("shape_id")> id;
            utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat;
            utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon;
            utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq;
        };

        struct MemoryMapConfig {
            const std::string_view coordinates_file;
            const std::string_view metadata_file;
            // const std::string_view& ids_file;
        };


        static constexpr ShapePoint from_shape(const Shape& shape) {
            return ShapePoint{
                shape.id.val(),
                {
                    helper::double_to_fix(shape.lat.val()),
                    helper::double_to_fix(shape.lon.val()),
                },
                shape.seq.val(),
            };
        }
    };

    using shape_data_t = helper::mm_vecvec<std::size_t, ShapePoint::Coordinate>;

    inline shape_data_t create_shape_memory_map(const ShapePoint::MemoryMapConfig& config,
            const cista::mmap::protection mode = cista::mmap::protection::READ) {
        return shape_data_t{
            cista::basic_mmap_vec<ShapePoint::Coordinate, std::size_t>{
                cista::mmap{config.coordinates_file.data(), mode}},
            cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{
                config.metadata_file.data(), mode}}};
    }

    inline shape_id_vec shape_load_map(const std::string_view& data, shape_data_t& map) {
        shape_id_type last_id;
        auto bucket = map.add_back_sized(0u);
        shape_id_vec ids;

        auto store_to_map = [&bucket, &map, &ids, &last_id](const auto& point) {
            if (last_id.empty()) {
                ids.push_back(point.id);
                last_id = point.id;
            } else if (last_id != point.id) {
                ids.push_back(point.id);
                bucket = map.add_back_sized(0u);
                last_id = point.id;
            }
            bucket.push_back(point.coordinate);
        };

        utl::line_range{utl::make_buf_reader(data, utl::noop_progress_consumer{})}
            | utl::csv<ShapePoint::Shape>()
            | utl::transform([&](ShapePoint::Shape const& shape) {
                return ShapePoint::from_shape(shape);
              })
            | utl::for_each(store_to_map);

        return ids;
    }

}
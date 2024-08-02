#pragma once

#include <filesystem>
#include <vector>
#include <unordered_map>

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"
#include "cista/mmap.h"
#include "geo/latlng.h"
#include "utl/parser/csv_range.h"

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

    class InvalidShapesFormat final : public std::runtime_error {
    public:
        InvalidShapesFormat(const std::string& msg);
    };

    class ShapeMap {
    public:
        using key_type = std::string;
        struct Paths {
            std::filesystem::path id_file;
            std::filesystem::path shape_data_file;
            std::filesystem::path shape_metadata_file;
        };

        ShapeMap(const std::string_view, const Paths&);
        ShapeMap(const Paths&);
        size_t size() const ;
        bool contains(const key_type&) const;
        std::vector<geo::latlng> at(const key_type&) const;
        static void write_shapes(const std::string_view, const Paths&);
    private:
        using shape_coordinate_type = std::remove_const<decltype(helper::coordinate_precision)>::type;
        struct Coordinate {
            shape_coordinate_type lat, lon;
            bool operator==(const Coordinate& other) const = default;
        };
        using shape_data_t = helper::mm_vecvec<std::size_t, ShapeMap::Coordinate>;
        using id_vec_t = std::vector<key_type>;
        using id_map_t = std::unordered_map<key_type, size_t>;

        ShapeMap(std::pair<shape_data_t, id_vec_t>);
        static std::pair<shape_data_t, id_vec_t> create_files(const std::string_view data, const Paths&);
        static std::pair<shape_data_t, id_vec_t> load_files(const Paths&);
        static shape_data_t create_memory_map(const Paths&, const cista::mmap::protection = cista::mmap::protection::READ);
        static auto create_id_memory_map(const std::filesystem::path&, const cista::mmap::protection = cista::mmap::protection::READ);
        static id_vec_t load_shapes(const std::string_view, shape_data_t&);
        static void store_ids(const id_vec_t&, const std::filesystem::path&);
        static id_vec_t load_ids(const std::filesystem::path&);
        static id_map_t id_vec_to_map(const id_vec_t&);
        static std::vector<geo::latlng> transform_coordinates(const auto&);

        const shape_data_t shape_map_;
        const id_map_t id_map_;

        struct ShapePoint {
            const key_type id;
            const Coordinate coordinate;
            const size_t seq;
            struct Entry {
                utl::csv_col<ShapeMap::key_type, UTL_NAME("shape_id")> id;
                utl::csv_col<double, UTL_NAME("shape_pt_lat")> lat;
                utl::csv_col<double, UTL_NAME("shape_pt_lon")> lon;
                utl::csv_col<size_t, UTL_NAME("shape_pt_sequence")> seq;
            };
            static constexpr ShapePoint from_entry(const Entry&);
        };
    };

}
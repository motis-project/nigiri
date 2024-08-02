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

    InvalidShapesFormat::InvalidShapesFormat(const std::string& msg) : std::runtime_error{msg} {}

    ShapeMap::ShapeMap(const std::string_view data, const Paths& paths) : ShapeMap(create_files(data, paths)) {}

    ShapeMap::ShapeMap(const Paths& paths) : ShapeMap(load_files(paths)) {}

    ShapeMap::ShapeMap(std::pair<shape_data_t, id_vec_t> p) : shape_map_{std::move(p.first)}, id_map_{id_vec_to_map(p.second)} {}

    size_t ShapeMap::size() const {
        return id_map_.size();
    }

    bool ShapeMap::contains(const key_type& id) const {
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

    ShapeMap::shapes_t ShapeMap::at(const key_type& id) const {
        auto offset = id_map_.at(id);
        return transform_coordinates(shape_map_.at(offset));
    }

    void ShapeMap::write_shapes(const std::string_view data, const Paths& paths) {
        create_files(data, paths);
    }

    std::pair<ShapeMap::shape_data_t, ShapeMap::id_vec_t> ShapeMap::create_files(const std::string_view data, const Paths& paths) {
        shape_data_t mmap{create_memory_map(paths, cista::mmap::protection::WRITE)};
        id_vec_t ids{load_shapes(data, mmap)};
        store_ids(ids, paths.id_file);
        return std::make_pair(std::move(mmap), std::move(ids));
    }

    std::pair<ShapeMap::shape_data_t, ShapeMap::id_vec_t> ShapeMap::load_files(const Paths& paths) {
        shape_data_t mmap{create_memory_map(paths)};
        return std::make_pair(std::move(mmap), load_ids(paths.id_file));
    }

    ShapeMap::shape_data_t ShapeMap::create_memory_map(const Paths& paths, const cista::mmap::protection mode) {
        return shape_data_t{
            cista::basic_mmap_vec<Coordinate, std::size_t>{
                cista::mmap{paths.shape_data_file.native().data(), mode}},
            cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{
                paths.shape_metadata_file.native().data(), mode}}};
    }

    auto ShapeMap::create_id_memory_map(const std::filesystem::path& path, const cista::mmap::protection mode) {
        return cista::mmap_vec<key_type::value_type>{cista::mmap{path.native().data(), mode}};
    }

    ShapeMap::id_vec_t ShapeMap::load_shapes(const std::string_view data, shape_data_t& mmap) {
        struct Current {
            key_type id;
            shape_data_t::bucket bucket;
            size_t last_seq{};
            size_t offset{};
        };
        struct BucketInfo {
            size_t offset{};
            size_t last_seq{};
        };
        struct {
            std::optional<Current> current;
            std::unordered_map<key_type, BucketInfo> info;
            id_vec_t ids;
        } state;

        auto add_new_bucket = [&mmap, &state](const ShapePoint& point) {
            auto bucket = mmap.add_back_sized(0u);
            bucket.push_back(point.coordinate);
            state.current.emplace(Current{
                point.id,
                bucket,
                point.seq,
                state.ids.size(),
            });
            state.ids.push_back(point.id);
        };
        auto append_to_bucket = [&mmap](BucketInfo& info, const ShapePoint& point) {
            mmap.at(info.offset).push_back(point.coordinate);
            info.last_seq = point.seq;
        };
        auto append_to_current_bucket = [](Current& current, const ShapePoint& point) {
            if (current.last_seq >= point.seq) {
                throw InvalidShapesFormat(std::format("Non monotonic sequence for shape_id '{}': Sequence number {} followed by {}", point.id, current.last_seq, point.seq));
            }
            current.bucket.push_back(point.coordinate);
            current.last_seq = point.seq;
        };
        auto archive_bucket = [&state](const Current& current) {
            state.info.insert({current.id, {current.offset, current.last_seq}});
        };
        auto functions = std::make_tuple(
            add_new_bucket,
            append_to_bucket,
            append_to_current_bucket,
            archive_bucket
        );

        auto store_to_map = [&functions, &state](const ShapePoint point) {
            if (state.current.has_value()) {
                auto& current = state.current.value();
                if (current.id == point.id) {
                    std::get<2>(functions)(current, point);
                } else if (auto found = state.info.find(point.id); found != state.info.end()) {
                    std::get<1>(functions)(found->second, point);
                } else {
                    std::get<3>(functions)(current);
                    std::get<0>(functions)(point);
                }
            } else {
                std::get<0>(functions)(point);
            }
        };

        utl::line_range{utl::make_buf_reader(data, utl::noop_progress_consumer{})}
            | utl::csv<ShapePoint::Entry>()
            | utl::transform([&](ShapePoint::Entry const& entry) {
                return ShapePoint::from_entry(entry);
              })
            | utl::for_each(store_to_map);

        return std::move(state.ids);
    }
    void ShapeMap::store_ids(const id_vec_t& ids, const std::filesystem::path& path) {
        auto storage{create_id_memory_map(path, cista::mmap::protection::WRITE)};
        for (auto id : ids) {
            storage.insert(storage.end(), id.begin(), id.end());
            storage.push_back('\0');
        }
    }

    ShapeMap::id_vec_t ShapeMap::load_ids(const std::filesystem::path& path) {
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

    ShapeMap::id_map_t ShapeMap::id_vec_to_map(const id_vec_t& ids) {
        id_map_t map;
        for (auto [pos, id] : std::ranges::enumerate_view(ids)) {
            map.insert({id, pos});
        }
        return map;
    }

    ShapeMap::shapes_t ShapeMap::transform_coordinates(const auto& shape) {
        auto coordinates = shape
            | std::views::transform([](const Coordinate& c) {
                return geo::latlng{helper::fix_to_double(c.lat), helper::fix_to_double(c.lon)};
            })
        ;
        return shapes_t{coordinates.begin(), coordinates.end()};
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

    ShapeMap::Iterator& ShapeMap::Iterator::operator++() {
        ++it;
        return *this;
    }

    ShapeMap::Iterator ShapeMap::Iterator::operator++(int) {
        auto tmp(*this);
        ++(*this);
        return tmp;
    }

    ShapeMap::shapes_t ShapeMap::Iterator::operator*() const {
        return shapes->at(it->first);
    }
}

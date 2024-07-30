#pragma once
// Duplicated from libosmium and osr


#include <array>
#include <cstdint>
#include <cmath>

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"

template <typename Columns>
class CsvReader {
private:
    using header_t = std::array<utl::column_idx_t, utl::MAX_COLUMNS>;
    struct Iterator {
        Iterator(const std::string_view data) : data_{data}, offset_{0u} {}
        bool operator==(Iterator& other) {
            return offset_ != other.offset_;
        }
        const std::string_view data_;
        const size_t offset_;
    };
    struct Sentinel {
        bool operator==(Iterator&) {
            return false;
        }
    };
    static header_t read_header(const std::string_view data) {
        auto offset = data.find_first_of("\r\n");  // Create string to ensure trailing '\0' byte
        return utl::read_header<Columns>(utl::cstr(data.data(), offset));
        // const std::string header{data.substr(0, data.find_first_of("\r\n"))};  // Create string to ensure trailing '\0' byte
        // // std::cout << std::format("Header: '{}'", header.data()) << std::endl;
        // return utl::read_header<Columns>(header.data());
    }
public:
    // CsvReader(const std::string_view data) : data_{data}, header_{read_header(data)} {}
    // CsvReader(const std::string_view data, const header_t columns) :
    CsvReader(const std::string_view data) :
        data_{data},
        header_{read_header(data)} {
            std::cout << "Debug: Header: '" << static_cast<int>(header_.at(1)) << "'" << std::endl;
        }
    Iterator begin() const {
        return Iterator(data_);
    }
    Sentinel end() const {
        return Sentinel{};
    }

private:
    const std::string_view data_;
    const header_t header_;
};

template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename K, typename V, typename SizeType = cista::base_t<K>>
using mm_vecvec = cista::basic_vecvec<K, mm_vec<V>, mm_vec<SizeType>>;


constexpr int32_t coordinate_precision{10000000};

constexpr int32_t double_to_fix(const double c) noexcept {
    return static_cast<int32_t>(std::round(c * coordinate_precision));
}

constexpr double fix_to_double(const int32_t c) noexcept {
    return static_cast<double>(c) / coordinate_precision;
}
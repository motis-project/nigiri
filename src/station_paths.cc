#include "nigiri/station_paths.h"

#include <algorithm>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

namespace nigiri {

station_paths::station_paths(std::filesystem::path path,
                             cista::mmap::protection const mode)
    : mode_{mode},
      p_{[&]() {
        fs::create_directories(path);
        return std::move(path);
      }()},
      nodes_{mm("station_nodes.bin")},
      pieces_{mm("station_pieces.bin")},
      paths_{mm("station_paths.bin")},
      path_pieces_{mm_vec<station_piece_idx_t>{mm("station_path_pieces.bin")},
                   mm_vec<std::uint64_t>{mm("station_path_pieces_idx.bin")}},
      entrances_{mm("station_entrances.bin")},
      entrance_paths_{mm("station_entrance_paths.bin")},
      entrance_path_pieces_{
          mm_vec<station_piece_idx_t>{mm("station_entrance_path_pieces.bin")},
          mm_vec<std::uint64_t>{mm("station_entrance_path_pieces_idx.bin")}},
      entrance_only_{mm("station_entrance_only.bin")},
      strings_{mm_vec<char>{mm("station_strings.bin")},
               mm_vec<std::uint64_t>{mm("station_strings_idx.bin")}} {}

cista::mmap station_paths::mm(char const* file) {
  return cista::mmap{(p_ / file).generic_string().c_str(), mode_};
}

std::optional<station_path_idx_t> station_paths::find(
    location_idx_t const from,
    location_idx_t const to,
    footpath_restriction const& r) const {
  auto const it = std::lower_bound(
      begin(paths_), end(paths_), std::pair{from, to},
      [](station_path const& p, std::pair<location_idx_t, location_idx_t> x) {
        return std::tie(p.from_, p.to_) < std::tie(x.first, x.second);
      });
  auto best = std::optional<station_path_idx_t>{};
  for (auto i = it; i != end(paths_) && i->from_ == from && i->to_ == to; ++i) {
    if (i->restriction_ == r &&
        (!best.has_value() ||
         i->walk_seconds_ + i->fixed_seconds_ <
             paths_[*best].walk_seconds_ + paths_[*best].fixed_seconds_)) {
      best = station_path_idx_t{
          static_cast<std::uint32_t>(std::distance(begin(paths_), i))};
    }
  }
  return best;
}

std::span<station_entrance_path const> station_paths::entrance_paths(
    station_entrance_idx_t const e, bool const is_exit) const {
  auto const [lb, ub] = std::equal_range(
      begin(entrance_paths_), end(entrance_paths_), std::pair{e, is_exit},
      [](auto const& a, auto const& b) {
        auto const key = [](auto const& x) {
          if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                                       station_entrance_path>) {
            return std::pair{x.entrance_, x.is_exit_};
          } else {
            return x;
          }
        };
        return key(a) < key(b);
      });
  return {lb, ub};
}

station_entrance_path_idx_t station_paths::get_idx(
    station_entrance_path const& p) const {
  return station_entrance_path_idx_t{
      static_cast<std::uint32_t>(&p - &*begin(entrance_paths_))};
}

bool station_paths::is_entrance_only(location_idx_t const l) const {
  return std::binary_search(begin(entrance_only_), end(entrance_only_), l);
}

std::string_view station_paths::get_string(station_string_idx_t const i) const {
  return i == station_string_idx_t::invalid() ? std::string_view{}
                                              : strings_[i].view();
}

}  // namespace nigiri

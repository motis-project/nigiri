#pragma once

#include "cista/reflection/printable.h"

#include "nigiri/types.h"

namespace nigiri {

struct footpath {
  using value_type = location_idx_t::value_t;

  footpath() = default;

  footpath(location_idx_t::value_t const val) {
    std::memcpy(this, &val, sizeof(value_type));
  }

  footpath(location_idx_t const target, u8_minutes const duration)
      : target_{target}, duration_{duration.count()} {
    assert(target < kMaxLocations);
  }

  location_idx_t target() const { return location_idx_t{target_}; }
  std::uint8_t duration_uint() const { return duration_; }
  duration_t duration() const { return duration_t{duration_uint()}; }

  location_idx_t::value_t value() const {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  friend std::ostream& operator<<(std::ostream& out, footpath const& fp) {
    return out << "(" << fp.target() << ", " << fp.duration() << ")";
  }

  location_idx_t::value_t target_ : 24;
  location_idx_t::value_t duration_ : 8;
};

template <std::size_t NMaxTypes>
constexpr auto static_type_hash(footpath const*,
                                cista::hash_data<NMaxTypes> h) noexcept {
  return h.combine(cista::hash("nigiri::footpath"));
}

template <typename Ctx>
inline void serialize(Ctx&, footpath const*, cista::offset_t const) {}

template <typename Ctx>
inline void deserialize(Ctx const&, footpath*) {}

}  // namespace nigiri
#pragma once

#include "fmt/ostream.h"

#include "utl/verify.h"

#include "cista/reflection/printable.h"

#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri {

struct footpath {
  using value_type = location_idx_t::value_t;
  static constexpr auto const kTotalBits = 8 * sizeof(value_type);
  static constexpr auto const kTargetBits = 22U;
  static constexpr auto const kDurationBits = kTotalBits - kTargetBits;
  static constexpr auto const kMaxDuration = duration_t{
      std::numeric_limits<location_idx_t::value_t>::max() >> kTargetBits};
  static constexpr auto const kMaxTarget =
      std::numeric_limits<location_idx_t::value_t>::max() >> kDurationBits;

  footpath() = default;

  footpath(location_idx_t::value_t const val) {
    std::memcpy(this, &val, sizeof(value_type));
  }

  footpath(location_idx_t const target, duration_t const duration)
      : target_{target},
        duration_{static_cast<value_type>(
            (duration > kMaxDuration ? kMaxDuration : duration).count())} {
    utl::verify(to_idx(target) < kMaxTarget, "station index overflow");
  }

  static auto cmp_by_duration() {
    return [](footpath const& a, footpath const& b) {
      return a.duration() < b.duration();
    };
  }

  location_idx_t target() const { return location_idx_t{target_}; }
  duration_t duration() const { return duration_t{duration_}; }

  location_idx_t::value_t value() const {
    return *reinterpret_cast<location_idx_t::value_t const*>(this);
  }

  friend std::ostream& operator<<(std::ostream& out, footpath const& fp) {
    return out << "(" << fp.target() << ", " << fp.duration() << ")";
  }

  bool operator==(footpath const& o) const {
    return target_ == o.target_ && duration_ == o.duration_;
  }
  auto operator<=>(footpath const& o) const {
    return std::tie(target_, duration_) <=> std::tie(o.target_, o.duration_);
  }

  location_idx_t::value_t target_ : kTargetBits;
  location_idx_t::value_t duration_ : kDurationBits;
};

static_assert(std::three_way_comparable<footpath>);

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

template <>
struct fmt::formatter<nigiri::footpath> : ostream_formatter {};
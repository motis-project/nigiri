#pragma once

#include <iosfwd>

#include "nigiri/types.h"

#include "utl/verify.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

using segment_idx_t = cista::strong<std::uint32_t, struct _segment_idx>;

using tb_bitfield_idx_t = cista::strong<std::uint32_t, struct _tb_bitfield_idx>;

struct transfer {
  void set_day_offset(std::int8_t const o) {
    utl::verify(o >= -8 && o < 8, "invalid day offset  {}", o);
    day_offset_ = static_cast<std::uint16_t>(o + 8);
  }

  inline std::int8_t get_day_offset() const {
    return static_cast<std::int8_t>(day_offset_) - 8;
  }

  template <typename Ctx>
  friend void serialize(Ctx&, transfer const*, cista::offset_t) {}

  template <typename Ctx>
  friend void deserialize(Ctx const&, transfer*) {}

  template <std::size_t NMaxTypes>
  constexpr friend auto static_type_hash(transfer const*,
                                         cista::hash_data<NMaxTypes> h) {
    return h.combine(cista::hash("tb_data::transfer::v1"));
  }

  segment_idx_t to_segment_;
  tb_bitfield_idx_t traffic_days_;
  route_idx_t route_;
  std::uint16_t transport_offset_;
  std::uint16_t to_segment_offset_ : 12;

private:
  // Shift amount between trip t and trip u:
  // This includes:
  //   - (positive) how many times did t cross midnight until the transfer
  //   - (negative) how many times did u cross midnight until the transfer
  //   - (+1) does the transfer itself cross midnight
  std::uint16_t day_offset_ : 4;
};

static_assert(sizeof(transfer) == 16);

struct tb_data {
  interval<segment_idx_t> get_segment_range(transport_idx_t const t) const {
    return {transport_first_segment_[t], transport_first_segment_[t + 1]};
  }

  void print(std::ostream&, timetable const&) const;

  profile_idx_t prf_idx_;
  vector_map<transport_idx_t, segment_idx_t> transport_first_segment_;
  vecvec<segment_idx_t, transfer> segment_transfers_;
  vector_map<segment_idx_t, transport_idx_t> segment_transports_;
  vector_map<tb_bitfield_idx_t, bitfield> bitfields_;
};

}  // namespace nigiri::routing::tb
#pragma once

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/service_expansion/ref_service.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void expand_traffic_days(service const& s,
                         bitfield_map_t const& bitfields,
                         Fn&& consumer) {
  // Transform section bitfield indices into concrete bitfields.
  auto section_bitfields =
      utl::to_vec(s.sections_, [&](service::section const& section) {
        auto const it = bitfields.find(section.traffic_days_.at(0));
        utl::verify(it != end(bitfields), "bitfield {} not found",
                    section.traffic_days_.at(0));
        return it->second.first;
      });

  // Checks that all section bitfields are disjunctive and calls consumer.
  bitfield consumed_traffic_days;
  auto const check_and_consume = [&](ref_service&& x) {
    utl::verify((x.split_info_.traffic_days_ & consumed_traffic_days).none(),
                "traffic days of service {} are not disjunctive:\n"
                "    sub-sections: {}\n"
                "already consumed: {}",
                x.ref_.origin_, x.split_info_.traffic_days_,
                consumed_traffic_days);
    consumed_traffic_days |= x.split_info_.traffic_days_;
    consumer(std::move(x));
  };

  // Removes the set bits from the section bitfields (for further iterations)
  // and writes a new services containing the specified sections [start, pos[.
  auto const consume_and_remove = [&](unsigned const start, unsigned const pos,
                                      bitfield const& current) {
    if (current.any() && start < pos) {
      auto const not_current = ~current;
      for (unsigned i = start; i < pos; ++i) {
        section_bitfields[i] &= not_current;
      }
      assert(pos >= 1);
      check_and_consume(ref_service{s, split_info{current, {start, pos}}});
    }
  };

  // Function splitting services with uniform traffic day bitfields.
  auto const split = [&](unsigned const start) {
    auto b = section_bitfields[start];
    for (auto i = start + 1; i != section_bitfields.size(); ++i) {
      auto const next_b = b & section_bitfields[i];
      if (next_b.none()) {
        consume_and_remove(start, i, b);
        return;
      }
      b = next_b;
    }
    consume_and_remove(start, section_bitfields.size(), b);
  };

  for (auto i = 0U; i < section_bitfields.size(); ++i) {
    if (section_bitfields[i].none()) {
      return;
    }
    split(i);
  }
}

}  // namespace nigiri::loader::hrd

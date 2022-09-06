#pragma once

#include "utl/to_vec.h"

#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/stamm/bitfield.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void expand_traffic_days(service_store const& store,
                         service_idx_t const s_idx,
                         stamm& st,
                         Fn&& consumer) {
  auto const& s = store.get(s_idx);

  // Transform section bitfield indices into concrete bitfields.
  auto section_bitfields =
      s.begin_to_end_info_.traffic_days_.has_value()
          ? std::vector<bitfield>{s.stops_.size() - 1U,
                                  st.resolve_bitfield(
                                      s.begin_to_end_info_.traffic_days_
                                          .value())}
          : utl::to_vec(s.sections_, [&](service::section const& section) {
              assert(section.traffic_days_.has_value());
              return st.resolve_bitfield(section.traffic_days_.value());
            });

  // Checks that all section bitfields are disjunctive and calls consumer.
  bitfield consumed_traffic_days;
  auto const check_and_consume = [&](ref_service&& x) {
    utl::verify((x.split_info_.traffic_days_ & consumed_traffic_days).none(),
                "traffic days of service {} are not disjunctive:\n"
                "    sub-sections: {}\n"
                "already consumed: {}",
                x.origin(store), x.split_info_.traffic_days_,
                consumed_traffic_days);
    consumed_traffic_days |= x.split_info_.traffic_days_;
    consumer(std::move(x));
  };

  // Removes the set bits from the section bitfields (for further iterations)
  // and writes a new services containing the specified sections [start, pos[.
  auto const consume_and_remove = [&](std::size_t const start,
                                      std::size_t const pos,
                                      bitfield const& current) {
    if (current.any() && start < pos) {
      auto const not_current = ~current;
      for (auto i = start; i < pos; ++i) {
        section_bitfields[i] &= not_current;
      }
      assert(pos >= 1);
      check_and_consume(ref_service{s_idx, split_info{current, {start, pos}}});
    }
  };

  // Function splitting services with uniform traffic day bitfields.
  auto const split = [&](std::size_t const start) {
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

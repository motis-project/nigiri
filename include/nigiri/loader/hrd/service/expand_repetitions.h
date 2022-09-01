#pragma once

#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/service/service_store.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void expand_repetitions(service_store const& store,
                        ref_service const& s,
                        Fn&& consumer) {
  for (auto rep = 0U; rep <= store.get(s.ref_).num_repetitions_; ++rep) {
    consumer(ref_service{s, rep});
  }
}

}  // namespace nigiri::loader::hrd

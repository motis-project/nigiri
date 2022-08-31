#pragma once

#include "ref_service.h"

namespace nigiri::loader::hrd {

template <typename Fn>
void expand_repetitions(ref_service const& s, Fn&& consumer) {
  for (int rep = 0; rep <= s.ref_.num_repetitions_; ++rep) {
    consumer(ref_service{s, rep});
  }
}

}  // namespace nigiri::loader::hrd

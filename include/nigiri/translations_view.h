#pragma once

#include <ranges>

#include "nigiri/timetable.h"

namespace nigiri {

inline auto get_translation_view(timetable const& tt,
                                 translation_idx_t const t) {
  namespace sv = std::views;
  return sv::zip(
      tt.translation_language_[t],
      tt.translations_[t] | sv::transform([](auto&& y) { return y.view(); }));
}

}  // namespace nigiri
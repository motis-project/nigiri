#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace nigiri {

inline void prefetch([[maybe_unused]] void const* addr) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<char const*>(addr), _MM_HINT_T0);
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
  __prefetch(addr);
#endif
}

}  // namespace nigiri

#include "nigiri/routing/clasz_mask.h"

namespace nigiri::routing {

std::string to_str(clasz_mask_t const x) {
  auto s = std::string{};
  s += ", x=" + std::to_string(x);
  for (auto i = std::underlying_type_t<clasz>{0U};
       i != sizeof(clasz_mask_t) * 8; ++i) {
    auto const allowed = is_allowed(x, clasz{i});
    s.insert(0, 1, allowed ? '1' : '0');
  }
  return s;
}

}  // namespace nigiri::routing
#pragma once

#include "cista/strong.h"

#include "nigiri/loader/hrd/service/service.h"

namespace nigiri::loader::hrd {

using service_idx_t = cista::strong<std::uint32_t, service>;

struct service_store {
  service_idx_t add(service&& s) {
    auto const idx = service_idx_t{
        static_cast<cista::base_t<service_idx_t>>(services_.size())};
    services_.emplace_back(std::move(s));
    return idx;
  }
  service const& get(service_idx_t const idx) const {
    return services_.at(to_idx(idx));
  }
  void clear() { services_.clear(); }

private:
  std::vector<service> services_;
};

}  // namespace nigiri::loader::hrd
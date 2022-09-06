#pragma once

#include <cinttypes>
#include <functional>

namespace nigiri::loader::hrd {

using progress_update_fn =
    std::function<void(std::size_t /* bytes processed */)>;

}  // namespace nigiri::loader::hrd
#pragma once

#include "nigiri/common/interval.h"

namespace nigiri {

template <typename Timepoint>
inline constexpr auto const kMaxInterval = interval<Timepoint>{
    Timepoint{typename Timepoint::duration{
        std::numeric_limits<typename Timepoint::duration::rep>::min()}},
    Timepoint{typename Timepoint::duration{
        std::numeric_limits<typename Timepoint::duration::rep>::max()}}};

}  // namespace nigiri

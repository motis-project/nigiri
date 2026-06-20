#pragma once

#define MAX_DAYS 384

#define BITFIELD_IDX_BITS 25U
#define TRANSPORT_IDX_BITS 26U
#define STOP_IDX_BITS 10U
#define DAY_IDX_BITS 9U
#define NUM_TRANSFERS_BITS 5U
#define DAY_OFFSET_BITS 3U

// the position of the query day in the day offset
#define QUERY_DAY_SHIFT 5

namespace nigiri::routing::tb {

constexpr auto const kTBMaxTravelTimeDays = 3U;
constexpr auto const kTBMaxDayOffset =
    std::int8_t{kTimetableOffset / 1_days + kTBMaxTravelTimeDays};

constexpr unsigned const kBitfieldIdxMax = 1U << BITFIELD_IDX_BITS;
constexpr unsigned const kTransportIdxMax = 1U << TRANSPORT_IDX_BITS;
constexpr unsigned const kStopIdxMax = 1U << STOP_IDX_BITS;
constexpr unsigned const kDayIdxMax = 1U << DAY_IDX_BITS;
constexpr unsigned const kNumTransfersMax = 1U << NUM_TRANSFERS_BITS;
constexpr unsigned const kDayOffsetMax = 1U << DAY_OFFSET_BITS;

}  // namespace nigiri::routing::tb
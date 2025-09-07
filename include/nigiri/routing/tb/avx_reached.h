#pragma once

#include <immintrin.h>

#include "nigiri/routing/tb/reached.h"

// #define NIGIRI_VERIFY_REACHED

namespace nigiri::routing::tb {

template <typename T>
inline std::array<T, sizeof(__m256i) / sizeof(T)> to_arr(__m256i value) {
  auto buf = std::array<T, sizeof(__m256i) / sizeof(T)>{};
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf.data()), value);
  return buf;
}

struct avx_pareto_set {
  void reset(std::uint16_t const max_segment_offset) {
    max_segment_offset_ = static_cast<short>(max_segment_offset);
    ks_ = _mm256_set1_epi16(INT16_MAX);
    transports_ = _mm256_set1_epi16(INT16_MAX);
    segment_offsets_ = _mm256_set1_epi16(INT16_MAX);
    if (overflow_) {
      [[unlikely]] overflow_->reset(max_segment_offset);
    }
  }

  inline void invalidate_dominated(__m256i const& k_b,
                                   __m256i const& t_b,
                                   __m256i const& s_b) {
    auto const dominated_mask = _mm256_xor_si256(
        // clang-format off
        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi16(k_b, ks_),
                _mm256_cmpgt_epi16(t_b, transports_)),
                _mm256_cmpgt_epi16(s_b, segment_offsets_)),
        // clang-format on
        _mm256_set1_epi16(0xFFFF)  // NEGATE
    );
    auto const invalid_value = _mm256_set1_epi16(INT16_MAX);
    ks_ = _mm256_blendv_epi8(ks_, invalid_value, dominated_mask);
    transports_ =
        _mm256_blendv_epi8(transports_, invalid_value, dominated_mask);
    segment_offsets_ =
        _mm256_blendv_epi8(segment_offsets_, invalid_value, dominated_mask);
  }

  template <bool Debug = false>
  inline void add(entry const& el) {
    auto const k_b = _mm256_set1_epi16(static_cast<short>(el.k_));
    auto const t_b = _mm256_set1_epi16(static_cast<short>(el.transport_));
    auto const s_b = _mm256_set1_epi16(static_cast<short>(el.segment_offset_));

    // ==============================
    // 1) Is the new entry dominated?
    // ------------------------------

    // Dominance:
    // !(ks_[i] > k
    //    || transports_[i] > transport
    //    || segment_offsets_[i] > segment_offset)
    // ==
    // ks_[i] <= k
    //    && transports_[i] <= transport
    //    && segment_offsets_[i] <= segment_offset
    auto const dominates_new_entry_mask = _mm256_xor_si256(
        // clang-format off
        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi16(ks_, k_b),
                _mm256_cmpgt_epi16(transports_, t_b)),
                _mm256_cmpgt_epi16(segment_offsets_, s_b)),
        // clang-format on
        _mm256_set1_epi16(0xFFFF)  // NEGATE
    );

    // Return if any existing dominated the new entry <=> any bit set in mask.
    if (!_mm256_testz_si256(dominates_new_entry_mask,
                            dominates_new_entry_mask)) {
      return;
    }

    // =====================================
    // 2) Are there any dominated entries?
    //    YES -> replace them with new entry
    //    NO  -> handle overflow
    // -------------------------------------

    // Same as above, just swapped operands.
    auto const dominated_mask = _mm256_xor_si256(
        // clang-format off
        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_cmpgt_epi16(k_b, ks_),
                _mm256_cmpgt_epi16(t_b, transports_)),
                _mm256_cmpgt_epi16(s_b, segment_offsets_)),
        // clang-format on
        _mm256_set1_epi16(0xFFFF)  // NEGATE
    );

    // Found at least one dominated entry.
    // Overwrite all dominated entries with the new entry.
    if (!_mm256_testz_si256(dominated_mask, dominated_mask)) {
      [[likely]];
      // Find the position of the first dominated element
      auto const mask_bits = _mm256_movemask_epi8(dominated_mask);
      auto const first_pos = static_cast<std::uint32_t>(
          __builtin_ctz(static_cast<std::uint32_t>(mask_bits)) / 2);

      // First, invalidate ALL dominated entries.
      auto const invalid_value = _mm256_set1_epi16(INT16_MAX);
      ks_ = _mm256_blendv_epi8(ks_, invalid_value, dominated_mask);
      transports_ =
          _mm256_blendv_epi8(transports_, invalid_value, dominated_mask);
      segment_offsets_ =
          _mm256_blendv_epi8(segment_offsets_, invalid_value, dominated_mask);

      // Replace first dominated entry.
      auto ks_array = std::bit_cast<std::array<int16_t, 16>>(ks_);
      auto transports_array =
          std::bit_cast<std::array<int16_t, 16>>(transports_);
      auto segments_array =
          std::bit_cast<std::array<int16_t, 16>>(segment_offsets_);

      ks_array[first_pos] = static_cast<int16_t>(el.k_);
      transports_array[first_pos] = static_cast<int16_t>(el.transport_);
      segments_array[first_pos] = static_cast<int16_t>(el.segment_offset_);

      ks_ = std::bit_cast<__m256i>(ks_array);
      transports_ = std::bit_cast<__m256i>(transports_array);
      segment_offsets_ = std::bit_cast<__m256i>(segments_array);

      if (overflow_ != nullptr) {
        [[unlikely]] overflow_->invalidate_dominated(k_b, t_b, s_b);
      }

      return;
    }

    // No dominated entry found. Overflow.
    if (overflow_ == nullptr) {
      reached_dbg("OVERFLOW\n");
      overflow_ = std::make_unique<avx_pareto_set>();
      overflow_->reset(static_cast<std::uint16_t>(max_segment_offset_));
    }
    overflow_->add(el);
  }

  template <bool Debug = false>
  inline std::uint16_t query(std::uint16_t const transport,
                             std::uint16_t const k) {
    auto const transport_broadcast =
        _mm256_set1_epi16(static_cast<short>(transport));
    auto const k_broadcast = _mm256_set1_epi16(static_cast<short>(k));

    // ks_ > k
    auto const k_gt_mask = _mm256_cmpgt_epi16(ks_, k_broadcast);

    // transports_ > transport
    auto const transport_gt_mask =
        _mm256_cmpgt_epi16(transports_, transport_broadcast);

    // (ks_ > k)   OR   (transports_ > transport)
    auto const gt_mask = _mm256_or_si256(k_gt_mask, transport_gt_mask);

    // Invert to get "less or equal":
    // !(ks_ > k || transports_ > transport)
    // = (ks_ <= k && transports_ <= transport)
    auto const lte_mask = _mm256_xor_si256(gt_mask, _mm256_set1_epi16(0xFFFF));

    // masked_segments[i] =
    //    (ks_[i] <= k && transports_[i] <= transport)     (<=> lte_mask[i])
    //        ? segment_offsets_[i]
    //        : 0xFFFF
    //
    // Example:
    // segments:      1 2 3 4
    // AND lte_mask:  F 0 F 0    (where k <= ks_  AND  transport <= transports_)
    //              = 1 0 3 0
    // OR gt_mask     0 F 0 F    (where k > ks_   OR   transport > transports_)
    // AND max_seg    0 5 0 5
    //              = 1 5 3 5
    auto const masked_segments = _mm256_or_si256(
        _mm256_and_si256(lte_mask, segment_offsets_),
        _mm256_and_si256(gt_mask, _mm256_set1_epi16(max_segment_offset_)));

    // =========================================
    // Horizontal minimum across masked_segments
    // -----------------------------------------

    // shuffle(1, 0, 3, 2) - read left to right (2, 3, 0, 1)
    // Swap lanes: (0, 1, 2, 3) => (2, 3, 0, 1)

    // One 64bit shuffle:
    // A      0 1 2 3 | 4 5 6 7  |  8 9 A B | C D E F
    // B      8 9 A B | C D E F  |  0 1 2 3 | 4 5 6 7
    // MIN1   0 1 2 3 | 4 5 6 7  |  0 1 2 3 | 4 5 6 7
    auto const perm1 =
        _mm256_permute4x64_epi64(masked_segments, _MM_SHUFFLE(1, 0, 3, 2));
    auto const min1 = _mm256_min_epu16(masked_segments, perm1);

    // Two parallel 32bit shuffles in the lower and upper halves:
    // MIN1   0 1 | 2 3 | 4 5 | 6 7  ||  0 1 | 2 3 | 4 5 | 6 7
    // PERM2  4 5 | 6 7 | 0 1 | 2 3  ||  4 5 | 6 7 | 0 1 | 2 3
    // MIN2   0 1 | 2 3 | 0 1 | 2 3  ||  0 1 | 2 3 | 0 1 | 2 3
    auto const perm2 = _mm256_shuffle_epi32(min1, _MM_SHUFFLE(1, 0, 3, 2));
    auto const min2 = _mm256_min_epu16(min1, perm2);

    // 16 bit shuffle in lower half.
    // Ignore higher half as it's the same.
    // MIN2   0 | 1 | 2 | 3
    // PERM3  2 | 3 | 0 | 1
    // MIN3   0 | 1 | 0 | 1
    auto const perm3 = _mm256_shufflelo_epi16(min2, _MM_SHUFFLE(1, 0, 3, 2));
    auto const min3 = _mm256_min_epu16(min2, perm3);

    // MIN3   0 | 1 | 0 | 1
    // PERM4  1 | 0 | 1 | 0
    // MIN4   0 | 0 | 0 | 0
    auto const perm4 = _mm256_shufflelo_epi16(min3, _MM_SHUFFLE(0, 1, 0, 1));
    auto const min4 = _mm256_min_epu16(min3, perm4);

    // Extract first.
    auto min = static_cast<std::uint16_t>(_mm256_extract_epi16(min4, 0));
    if (overflow_ != nullptr) {
      if constexpr (Debug) {
        fmt::println("OVERFLOW QUERY");
      }
      [[unlikely]] min = std::min(overflow_->query<Debug>(transport, k), min);
    }

    if constexpr (Debug) {
      fmt::println(
          "transport={}\n"
          "k={}\n"
          "transports={}\n"
          "ks={}\n"
          "segments={}\n"
          "k_gt_mask={}\n"
          "transport_gt_mask={}\n"
          "gt_mask={}\n"
          "lte_mask={}\n"
          "masked_segments={}\n"
          "perm1={}\n"
          "min1={}\n"
          "perm2={}\n"
          "min2={}\n"
          "perm3={}\n"
          "min3={}\n"
          "perm4={}\n"
          "min4={}\n"
          "min={}",
          transport, k, to_arr<short>(transports_), to_arr<short>(ks_),
          to_arr<short>(segment_offsets_), to_arr<short>(k_gt_mask),
          to_arr<short>(transport_gt_mask), to_arr<short>(gt_mask),
          to_arr<short>(lte_mask), to_arr<short>(masked_segments),
          to_arr<short>(perm1), to_arr<short>(min1), to_arr<short>(perm2),
          to_arr<short>(min2), to_arr<short>(perm3), to_arr<short>(min3),
          to_arr<short>(perm4), to_arr<short>(min4), min);
    }

    return min;
  }

  void print() {
    fmt::println(
        "transports={}\n"
        "ks={}\n"
        "segments={}\n",
        to_arr<short>(transports_), to_arr<short>(ks_),
        to_arr<short>(segment_offsets_));
    if (overflow_) {
      fmt::println("OVERFLOW:\n");
      overflow_->print();
    }
  }

  // 16 x 16bit
  __m256i ks_;
  __m256i transports_;
  __m256i segment_offsets_;
  std::unique_ptr<avx_pareto_set> overflow_;
  short max_segment_offset_;
};

struct avx_reached {
  explicit avx_reached(timetable const& tt, tb_data const& tbd)
      : tt_{tt}, tbd_{tbd}, data_{tt.n_routes()} {}

  void reset() {
    for (auto r = route_idx_t{0U}; r != tt_.n_routes(); ++r) {
      reached_dbg("RESET");
      data_[to_idx(r)].reset(
          static_cast<std::uint16_t>(tt_.route_location_seq_[r].size() - 1));
    }

#ifdef NIGIRI_VERIFY_REACHED
    verify_.reset();
#endif
  }

  inline void update(route_idx_t const r,
                     std::uint16_t const transport_offset,
                     std::uint16_t const segment_offset,
                     query_day_offset_t const query_day_offset,
                     std::uint8_t const k,
                     [[maybe_unused]] std::uint64_t& max_entries) {
    assert(query_day_offset >= 0 && query_day_offset < kTBMaxDayOffset);
    reached_dbg(
        "  reached update: k={}, r={}, dbg={}, trip={}, day={}, "
        "to_segment_offset={}",
        k, r, tt_.dbg(tt_.route_transport_ranges_[r][transport_offset]),
        tt_.transport_name(tt_.route_transport_ranges_[r][transport_offset]),
        query_day_offset, segment_offset);
    auto const transport = to_transport(transport_offset, query_day_offset);
    data_[to_idx(r)].add(
        {.transport_ = transport, .segment_offset_ = segment_offset, .k_ = k});

#ifdef NIGIRI_VERIFY_REACHED
    verify_.update(r, transport_offset, segment_offset, query_day_offset, k,
                   max_entries);
#endif
  }

  inline std::uint16_t query(route_idx_t const r,
                             std::uint16_t const transport_offset,
                             query_day_offset_t const query_day_offset,
                             std::uint8_t const k) {
    auto const transport = to_transport(transport_offset, query_day_offset);

#ifdef NIGIRI_VERIFY_REACHED
    auto const verify = verify_.query(r, transport_offset, query_day_offset, k);
    auto const uut = data_[to_idx(r)].query(transport, k);
    if (verify != uut) {
      data_[to_idx(r)].print();
      data_[to_idx(r)].template query<true>(transport, k);
      fmt::println("r={}, transport={}, k={}, verify={}, uut={}", r, transport,
                   k, verify, uut);
      std::terminate();
    }
#endif

    return data_[to_idx(r)].query(transport, k);
  }

  std::string to_str(day_idx_t const base, route_idx_t const r) const {
    auto const view = [](__m256i const& x) {
      return std::span{reinterpret_cast<unsigned short const*>(&x),
                       sizeof(__m256i) / sizeof(short)};
    };

    auto ss = std::stringstream{};
    ss << "route[" << r << "]=[\n";
    auto const& e = data_[to_idx(r)];
    for (auto const [k, transport, segment_offset] :
         utl::zip(view(e.ks_), view(e.transports_), view(e.segment_offsets_))) {
      if (k == INT16_MAX) {
        ss << "      (EMTPY, segment=" << segment_offset << "),\n";
        continue;
      }
      auto const t = tt_.route_transport_ranges_[r].from_ +
                     get_transport_offset(transport);
      auto const segment = tbd_.get_segment_range(t).from_ + segment_offset;
      auto const day = base + get_query_day(transport);

      ss << "      (" << k << ", " << segment_info{tt_, tbd_, segment, day}
         << "),\n";
    }
    ss << "  ]\n";

    return ss.str();
  }

  timetable const& tt_;
  tb_data const& tbd_;
  std::vector<avx_pareto_set> data_;

#ifdef NIGIRI_VERIFY_REACHED
  reached verify_{tt_, tbd_};
#endif
};

}  // namespace nigiri::routing::tb

#undef NIGIRI_VERIFY_REACHED
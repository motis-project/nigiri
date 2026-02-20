#pragma once

#include <cstddef>
#include <cstdint>
#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"
#include <algorithm>
#include <limits>
#include "utl/get_or_create.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {

static constexpr auto const kSawMetadataOffset = 3U;
static constexpr auto const kSawFieldLastSetBit = 0U;

struct traffic_days {
  vector_map<bitfield_idx_t, std::pair<bitfield, std::uint16_t>> bitfields_;
  hash_map<bitfield, bitfield_idx_t> bitfield_indices_;

  bitfield_idx_t get_or_create(bitfield t, std::uint16_t last_set_bit) {
    return utl::get_or_create(bitfield_indices_, t, [&]() {
      auto const r = bitfield_idx_t{bitfields_.size()};
      bitfields_.emplace_back(t, last_set_bit);
      return r;
    });
  }
};

template <saw_type SawType>
struct interleaved_saws;

template <saw_type SawType>
struct saw {

  static std::vector<tooth> of(duration_t const d) {
    utl::verify(SawType == saw_type::kConstant, "of not impl");
    auto out = std::vector<tooth>{};
    out.push_back({std::numeric_limits<std::int16_t>::max(), d,
                   bitfield_idx_t::invalid()});
    return out;
  }

  static std::uint16_t last_set_bit(bitfield const& t) {
    assert(kMaxDays % t.bits_per_block == 0);
    for (auto i = std::size_t{0U}; i != t.blocks_.size(); ++i) {
      auto const j = t.blocks_.size() - i - 1;
      if (t.blocks_[j] != 0U) {
        for (auto bit = std::size_t{0U}; bit != t.bits_per_block; ++bit) {
          if ((t.blocks_[j] & (cista::bitset<512>::block_t{1U}
                               << (t.bits_per_block - bit - 1))) != 0U) {
            return static_cast<std::uint16_t>(j * t.bits_per_block +
                                              t.bits_per_block - bit - 1);
          }
        }
      }
    }
    return 0U;
  }

  static std::uint16_t first_set_bit(bitfield const& t) {
    assert(kMaxDays % t.bits_per_block == 0);
    for (auto i = std::size_t{0U}; i != t.blocks_.size(); ++i) {
      if (t.blocks_[i] != 0U) {
        for (auto bit = std::size_t{0U}; bit != t.bits_per_block; ++bit) {
          if ((t.blocks_[i] & (cista::bitset<512>::block_t{1U} << bit)) != 0U) {
            return static_cast<std::uint16_t>(i * t.bits_per_block + bit);
          }
        }
      }
    }
    return 0U;
  }

  struct iterator {
    using difference_type = size_t;
    using value_type = tooth;
    using reference = tooth const&;
    using pointer = tooth const*;
    using iterator_category = std::bidirectional_iterator_tag;

    iterator& operator++() {
      ++pos_;
      if (pos_ >= s_.saw_.size()) {
        --day_offset_;
        pos_ = kSawMetadataOffset;
      }
      return *this;
    }

    iterator& operator+=(std::size_t offset) {
      if (s_.saw_.size() > kSawMetadataOffset) {
        auto const tmp = pos_ + offset - kSawMetadataOffset;
        pos_ = (tmp % s_.size()) + kSawMetadataOffset;
        day_offset_ -= tmp / s_.size();
      }
      return *this;
    }

    iterator& operator--() {
      if (pos_ <= kSawMetadataOffset) {
        ++day_offset_;
        pos_ = s_.saw_.size();
      }
      --pos_;
      return *this;
    }

    bool operator==(iterator const o) const {
      return s_.saw_.begin() == o.s_.saw_.begin() && pos_ == o.pos_ &&
             day_offset_ == o.day_offset_;
    }

    bool operator!=(iterator o) const { return !(*this == o); }

    tooth const* operator->() const { return std::addressof(operator*()); }

    tooth const& operator*() const { return s_.saw_[pos_]; }

    saw<SawType> s_{};
    size_t pos_{kSawMetadataOffset};
    std::int8_t day_offset_{};
  };

  iterator begin() const {
    utl::verify(saw<SawType>{saw_, traffic_days_}.valid(), "invalid saw {}",
                saw_.size());
    return iterator{
        saw<SawType>{saw_, traffic_days_}, {kSawMetadataOffset}, {}};
  }

  iterator end() const {
    return iterator{saw<SawType>{saw_, traffic_days_},
                    {kSawMetadataOffset},
                    static_cast<std::int8_t>(saw_.empty() ? 0 : -1)};
  }

  iterator begin(saw<SawType> const& ms) { return ms.begin(); }
  iterator end(saw<SawType> const& ms) { return ms.end(); }

  std::reverse_iterator<iterator> rbegin() const {
    return std::make_reverse_iterator(end());
  }

  std::reverse_iterator<iterator> rend() const {
    return std::make_reverse_iterator(begin());
  }

  std::reverse_iterator<iterator> rbegin(saw<SawType> const& ms) {
    return ms.rbegin();
  }
  std::reverse_iterator<iterator> rend(saw<SawType> const& ms) {
    return ms.rend();
  }

  bool is_constant() const {
    if constexpr (SawType == saw_type::kConstant) {
      return true;
    }
    if (saw_.size() == 1 &&
        saw_[0].mam_ == std::numeric_limits<std::int16_t>::max()) {
      return true;
    }
    return false;
  }

  size_t size() const {
    if (saw_.empty()) {
      return 0U;
    }
    if (is_constant()) {
      return 1U;
    }
    return saw_.size() - kSawMetadataOffset;
  }

  bool valid() const {
    return saw_.empty() || is_constant() || saw_.size() >= kSawMetadataOffset;
  }

  std::tuple<bool, size_t, size_t> _less(saw<SawType> const& b,
                                         bool const exact_true = false) const {
    if (saw_.empty()) {
      return {false, 0U, 0U};
    }
    if (b.saw_.empty()) {
      return {true, 0U, 0U};
    }
    if (is_constant() || b.is_constant()) {
      // TODO exact_true ?
      return {min() < b.min(), 0U, 0U};
    }
    auto const lsb = std::max(get_last_set_bit(), b.get_last_set_bit());
    auto const interleaved = interleaved_saws<SawType>{*this, b};

    for (auto b_it = interleaved.begin(); b_it != interleaved.end(); ++b_it) {
      if (b_it.is_a()) {
        continue;
      }
      if (b_it->travel_dur_ >= kChMaxEdgeTime) {
        continue;
      }
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(b_it->traffic_days_).first;

      auto a_it = begin();

      a_it += b_it.pos_a_ - kSawMetadataOffset;

      // std::cout << "b less " << std::endl;
      if constexpr (SawType != saw_type::kDay) {
        if (!exact_true) {
          --a_it;
          while (b_it->mam_ == a_it->mam_ &&
                 b_it->travel_dur_ == a_it->travel_dur_ &&
                 b_it.day_offset_ ==
                     a_it.day_offset_) {  // TODO return false immediately?
                                          // (power saw)
            --a_it;
          }
          ++a_it;
        }
      }
      // std::cout << "dom a: " << *a_it << " " << *b_it << std::endl;
      auto const r = _non_dominated(*b_it, a_it, &remaining_traffic_days, lsb);
      if (std::get<0>(r)) {
        std::cout << "nondom ct:" << std::get<1>(r) << " fsb:" << std::get<2>(r)
                  << " lsb:" << std::get<3>(r) << std::endl;
        return {false, a_it.pos_, b_it.pos_b_};
      }
    }
    return {true, 0U, 0U};
  }

  bool less(saw<SawType> const& b, bool const exact_true = false) const {
    auto const r = _less(b, exact_true);
    std::cout << "less " << std::get<0>(r) << " ";
    if (saw_.size() > 0 && b.saw_.size() > 0) {
      print_tooth(std::cout, saw_[std::get<1>(r)], traffic_days_);
      print_tooth(std::cout, b.saw_[std::get<2>(r)], b.traffic_days_);
    }
    std::cout << std::endl;
    // std::cout << *this << std::endl;
    // std::cout << b << std::endl;
    /*
    auto const rr = max() < b.min();
    auto const rrr = min() > b.max();
    if (!((!rr || std::get<0>(r)) && (!rrr || !std::get<0>(r)))) {
      std::cout << "less error" << saw_[std::get<1>(r)] << " "
                << b.saw_[std::get<2>(r)] << std::endl;
      std::cout << *this << std::endl;
      std::cout << b << std::endl;
      utl::fail("less error {} {} {} {} {}", r, min(), max(), b.min(),
                      b.max());
    }*/
    return std::get<0>(r);
  }

  friend bool operator<(saw<SawType> const& a, saw<SawType> const& b) {
    return a.less(b);
  }

  friend bool operator>(saw<SawType> const& a, saw<SawType> const& b) {
    return b < a;
  }

  friend bool operator<=(saw<SawType> const& a, saw<SawType> const& b) {
    return !(b < a);
  }

  bool leq(saw<SawType> const& b, bool const exact_false = false) const {
    return !(b.less(*this, exact_false));
  }

  friend bool operator>=(saw<SawType> const& a, saw<SawType> const& b) {
    return !(a < b);
  }

  friend bool operator==(saw<SawType> const& a, saw<SawType> const& b) {
    if (a.saw_.size() != b.saw_.size()) {
      return false;
    }
    for (auto i = a.is_constant() ? 0U : kSawMetadataOffset; i < a.saw_.size();
         ++i) {
      if (a.saw_[i].mam_ != b.saw_[i].mam_) {
        return false;
      }
      if (a.saw_[i].travel_dur_ != b.saw_[i].travel_dur_) {
        return false;
      }
      if (a.saw_[i].traffic_days_ != b.saw_[i].traffic_days_) {
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(saw<SawType> const& a, saw<SawType> const& b) {
    return !(a == b);
  }

  void static print_tooth(std::ostream& out,
                          tooth const& e,
                          traffic_days const& td) {
    out << e << " ct:"
        << (e.traffic_days_ != bitfield_idx_t::invalid()
                ? td.bitfields_.at(e.traffic_days_).first.count()
                : 0U)
        << " fsb:"
        << (e.traffic_days_ != bitfield_idx_t::invalid()
                ? first_set_bit(td.bitfields_.at(e.traffic_days_).first)
                : 0U)
        << " lsb:"
        << (e.traffic_days_ != bitfield_idx_t::invalid()
                ? last_set_bit(td.bitfields_.at(e.traffic_days_).first)
                : 0U)
        << " ";
  }

  friend std::ostream& operator<<(std::ostream& out, saw<SawType> const& a) {
    for (auto const& e : a.saw_) {
      print_tooth(out, e, a.traffic_days_);
      // << a.traffic_days_.bitfields_.at(e.traffic_days_).first << "\n";
    }
    return out;
  }

  std::pair<u16_minutes, size_t> _max() const {
    if (saw_.empty()) {
      return {u16_minutes{kMaxTravelTime.count()}, 0U};
    }
    if (is_constant()) {
      return {saw_[0].travel_dur_, 0U};
    }
    auto max = 0;
    auto max_tooth = size_t{};

    auto const empty_tooth = std::vector<tooth>{};
    auto const empty_saw = saw<SawType>{empty_tooth, traffic_days_};
    auto const interleaved = interleaved_saws<SawType>{
        *this, empty_saw};  // TODO impl single loop iterator

    for (auto b_it = interleaved.begin(); b_it != interleaved.end(); ++b_it) {
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(b_it->traffic_days_).first;
      auto day_offset = 0;
      auto a_it = b_it;
      // std::cout << "b max " << std::endl;
      while (true) {
        // std::cout << "a max " << std::endl;
        --a_it;

        auto const mam_diff =
            a_it->mam_ - b_it->mam_ + a_it.day_offset_ * 24 * 60;

        if constexpr (SawType == saw_type::kDay) {
          max = std::max(max, a_it->travel_dur_.count() + mam_diff);
          max_tooth = a_it.pos_a_;
          break;
        }
        if (a_it.day_offset_ > kChMaxEdgeTime / kChDay) {
          break;
        }
        if (day_offset != a_it.day_offset_) {
          remaining_traffic_days.set(
              traffic_days_.bitfields_.at(b_it->traffic_days_).second, false);
          remaining_traffic_days <<= 1U;
          day_offset = a_it.day_offset_;
        }
        if ((remaining_traffic_days &
             traffic_days_.bitfields_.at(a_it->traffic_days_)
                 .first)  // TODO templ, check really needed?
                .any()) {
          max = std::max(max, a_it->travel_dur_.count() + mam_diff);
          max_tooth = a_it.pos_a_;
          remaining_traffic_days &=
              ~traffic_days_.bitfields_.at(a_it->traffic_days_).first;
          if (remaining_traffic_days.none()) {
            break;
          }
        }
      }
    }
    if (max == 0) {
      return {u16_minutes{kMaxTravelTime.count()},
              max_tooth};  // TODO handle connections that only run on a single
                           // occasion, but multiple times (depending on
                           // proportion of loaded timetable -> kMax?)
    }
    return {u16_minutes{max}, max_tooth};
  }

  u16_minutes max() const { return _max().first; }

  std::pair<u16_minutes, u16_minutes> min_max_waiting_time() const {
    if (saw_.empty()) {
      return {u16_minutes{kMaxTravelTime.count()},
              u16_minutes{kMaxTravelTime.count()}};
    }
    if (is_constant()) {
      return {u16_minutes{0}, u16_minutes{0}};
    }
    auto min = static_cast<int>(kChMaxWaitingTime.count());
    auto max = 0;

    auto const empty_tooth = std::vector<tooth>{};
    auto const empty_saw = saw<SawType>{empty_tooth, traffic_days_};
    auto const interleaved = interleaved_saws<SawType>{
        *this, empty_saw};  // TODO impl single loop iterator

    for (auto b_it = interleaved.begin(); b_it != interleaved.end(); ++b_it) {
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(b_it->traffic_days_).first;
      auto day_offset = 0;
      auto a_it = b_it;
      // std::cout << "b max " << std::endl;
      while (true) {
        // std::cout << "a max " << std::endl;
        --a_it;

        auto const mam_diff =
            a_it->mam_ - b_it->mam_ + a_it.day_offset_ * 24 * 60;

        if constexpr (SawType == saw_type::kDay) {
          min = std::min(min, mam_diff);
          max = std::max(max, mam_diff);
          break;
        }
        if (a_it.day_offset_ > routing::kMaxTravelTime / 1_days) {
          break;
        }
        if (day_offset != a_it.day_offset_) {
          remaining_traffic_days.set(
              traffic_days_.bitfields_.at(b_it->traffic_days_).second, false);
          remaining_traffic_days <<= 1U;
          day_offset = a_it.day_offset_;
        }
        if ((remaining_traffic_days &
             traffic_days_.bitfields_.at(a_it->traffic_days_)
                 .first)  // TODO templ, check really needed?
                .any()) {
          min = std::min(min, mam_diff);
          max = std::max(max, mam_diff);
        }
      }
    }
    return {u16_minutes{min}, u16_minutes{max}};
  }

  u16_minutes max_travel_dur() const {
    if (saw_.empty()) {
      return u16_minutes{kMaxTravelTime.count()};
    }
    if (is_constant()) {
      return saw_.front().travel_dur_;
    }
    auto max = saw_[kSawMetadataOffset].travel_dur_.count();
    for (auto i = kSawMetadataOffset + 1U; i < saw_.size(); ++i) {
      max = std::max(max, saw_[i].travel_dur_.count());
    }
    return u16_minutes{max};
  }

  u16_minutes min() const {
    if (saw_.empty()) {
      return u16_minutes{kMaxTravelTime.count()};
    }
    if (is_constant()) {
      return saw_.front().travel_dur_;
    }
    auto min = saw_[kSawMetadataOffset].travel_dur_.count();
    for (auto i = kSawMetadataOffset + 1U; i < saw_.size(); ++i) {
      min = std::min(min, saw_[i].travel_dur_.count());
    }
    return u16_minutes{min};
  }

  saw<SawType> max(std::vector<tooth>& out,
                   saw_type to = SawType) const {  // TODO insitu
    utl::verify(out.empty(), "out not empty");
    simplify(saw<SawType>{{}, traffic_days_}, true, out);
    if (to == saw_type::kConstant) {
      auto tmp = saw<SawType>{out, traffic_days_}.max();
      out.clear();
      out.push_back({std::numeric_limits<std::int16_t>::max(), tmp,
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    return saw<SawType>{out, traffic_days_};
  }

  saw<SawType> min(std::vector<tooth>& out,
                   saw_type to = SawType) const {  // TODO insitu
    utl::verify(out.empty(), "out not empty");
    simplify(saw<SawType>{{}, traffic_days_}, false, out);
    if (to == saw_type::kConstant) {
      auto tmp = saw<SawType>{out, traffic_days_}.min();
      out.clear();
      out.push_back({std::numeric_limits<std::int16_t>::max(), tmp,
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    return saw<SawType>{out, traffic_days_};
  }

  template <typename Iterator>
  std::tuple<bool, std::uint16_t, std::uint16_t, std::uint16_t> _non_dominated(
      tooth const& t,
      Iterator& a_it,
      bitfield* remaining_traffic_days,
      std::uint16_t const lsb,
      bool const not_normalized = false,
      bool const day_lookahead = kChMaxEdgeTime / kChDay) const {
    auto day_offset = 0;

    while (true) {
      --a_it;
      if (a_it.day_offset_ > day_lookahead) {
        break;
      }
      auto const mam_diff = a_it->mam_ - t.mam_ + a_it.day_offset_ * 24 * 60;
      auto const remaining_travel_time =
          static_cast<std::int16_t>(t.travel_dur_.count()) - mam_diff;
      auto const is_infty_and_not_same_mam =
          a_it->travel_dur_.count() >= kChMaxEdgeTime.count() &&
          remaining_travel_time < kChMaxEdgeTime.count();

      if constexpr (SawType == saw_type::kDay) {
        if (remaining_travel_time >= a_it->travel_dur_.count() &&
            !is_infty_and_not_same_mam) {
          return {false, 0U, 0U, 0U};
        }
        if (!not_normalized) {
          return {true, 0U, 0U, 0U};
        }
      }
      if (is_infty_and_not_same_mam) {
        continue;
      }
      if (remaining_travel_time < 0) {  // TODO min const lb
        break;
      }
      if constexpr (SawType != saw_type::kDay) {
        if (day_offset != a_it.day_offset_) {
          // remaining_traffic_days->set(
          //     lsb, false);  // TODO this eats last_set_bits when
          //  non_dominated?, move inside dom check
          //  and set all remaining bits zero there?
          *remaining_traffic_days <<= 1U;
          day_offset = a_it.day_offset_;
        }
        if (remaining_travel_time >= a_it->travel_dur_.count()) {
          // std::cout << "bit sub " << *a_it << std::endl;
          //  TODO need to check conjunction? otherwise bit eating can set as
          //  dominated?
          *remaining_traffic_days &=
              ~traffic_days_.bitfields_.at(a_it->traffic_days_).first;
          auto remaining_count = remaining_traffic_days->count();
          if (remaining_count <=
              static_cast<unsigned>(day_offset)) {  // TODO efficient?
            for (auto i = 0U; i < static_cast<unsigned>(day_offset); ++i) {
              if (remaining_traffic_days->test(lsb + i + 1)) {
                --remaining_count;
              }
            }
            if (remaining_count == 0) {
              return {false, 0, 0, 0};
            }
          }
          // if (remaining_traffic_days->none()) {
          //   return false;
          // }
        }
      }
    }
    if constexpr (SawType == saw_type::kTrafficDaysPower) {
      *remaining_traffic_days >>= static_cast<unsigned>(
          day_offset);  // TODO shift other bitfields instead? – also
      // because this might eat last_set_bits???
    }

    return {true, remaining_traffic_days->count(),
            first_set_bit(*remaining_traffic_days),
            std::min(last_set_bit(*remaining_traffic_days), lsb)};
  }

  template <typename Iterator>
  bool non_dominated(tooth const& t,
                     Iterator& a_it,
                     bitfield* remaining_traffic_days,
                     std::uint16_t const lsb,
                     bool const not_normalized = false,
                     bool const day_lookahead = kChMaxEdgeTime / kChDay) const {
    return std::get<0>(_non_dominated(t, a_it, remaining_traffic_days, lsb,
                                      not_normalized, day_lookahead));
  }

  void init_metadata(std::vector<tooth>& out, std::uint16_t lsb) const {
    out.push_back({std::numeric_limits<std::int16_t>::max(), u16_minutes{lsb},
                   bitfield_idx_t::invalid()});
    out.push_back({std::numeric_limits<std::int16_t>::max(), u16_minutes{},
                   bitfield_idx_t::invalid()});
    out.push_back({std::numeric_limits<std::int16_t>::max(), u16_minutes{},
                   bitfield_idx_t::invalid()});
  }

  std::uint16_t get_last_set_bit() const {
    if (saw_.size() < kSawMetadataOffset) {
      return 0U;
    }
    return saw_[kSawFieldLastSetBit].travel_dur_.count();
  }

  void set_last_set_bit(std::vector<tooth>& out, std::uint16_t lsb) const {
    if (out.size() < kSawMetadataOffset) {
      return;
    }
    out[kSawFieldLastSetBit].travel_dur_ = u16_minutes(lsb);
  }

  saw<SawType> simplify(saw<SawType> const& other,
                        bool const max,
                        std::vector<tooth>& out) const {
    utl::verify(out.empty(), "out not empty");
    if (saw_.empty() && other.saw_.empty()) {
      return saw<SawType>{out, traffic_days_};
    }
    if (is_constant() || other.is_constant()) {
      auto const a_min = min();
      auto const b_min = other.min();
      if (!max || (is_constant() && other.is_constant()) ||
          (is_constant() && a_min <= b_min) ||
          (other.is_constant() && b_min <= a_min)) {
        out.push_back({std::numeric_limits<std::int16_t>::max(),
                       std::min(a_min, b_min), bitfield_idx_t::invalid()});
      } else {
        out.push_back({std::numeric_limits<std::int16_t>::max(),
                       std::max(a_min, b_min), bitfield_idx_t::invalid()});
      }
      // TODO horizontally cut tooths?

      return saw<SawType>{out, traffic_days_};
    }
    auto const lsb = std::max(get_last_set_bit(), other.get_last_set_bit());
    init_metadata(out, lsb);
    auto const interleaved = interleaved_saws<SawType>{*this, other};

    for (auto it = interleaved.begin(); it != interleaved.end(); ++it) {
      // std::cout << "simpli " << std::endl;
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(it->traffic_days_).first;

      auto that = it;
      if constexpr (SawType == saw_type::kTrafficDaysPower) {
        ++it;
        while (it != interleaved.end() && that->mam_ == it->mam_ &&
               that->travel_dur_ == it->travel_dur_) {
          remaining_traffic_days |=
              traffic_days_.bitfields_.at(it->traffic_days_).first;
          ++it;
        }
        --it;
      }
      auto ib = interleaved.begin();
      auto oe = saw<SawType>{out, traffic_days_}.end();
      if ((non_dominated(*that, ib, &remaining_traffic_days, lsb,
                         true)) &&  // TODO only look at other?
          (out.size() <= kSawMetadataOffset ||
           non_dominated(*that, oe, &remaining_traffic_days, lsb, false, 0))) {

        auto new_tooth = *that;
        if constexpr (SawType == saw_type::kTrafficDaysPower) {
          new_tooth.traffic_days_ =
              traffic_days_.get_or_create(remaining_traffic_days, lsb);
        }
        out.push_back(std::move(new_tooth));
      }
    }
    return saw<SawType>{out, traffic_days_};
  }

  saw<SawType> concat(saw<SawType> const& other,
                      bool const max,
                      std::vector<tooth>& out) const {
    utl::verify(out.empty(), "out not empty");
    if (saw_.empty() || other.saw_.empty()) {
      return saw<SawType>{out, traffic_days_};
    }
    if (other.is_constant()) {
      return concat_const(
          kForward, saw<saw_type::kConstant>{other.saw_, traffic_days_}, out);
    }
    if (is_constant()) {
      return other.concat_const(
          kReverse, saw<saw_type::kConstant>{saw_, traffic_days_}, out);
    }
    auto const_min_other = 0;
    if (SawType == saw_type::kTrafficDays && !max) {  // TODO constexpr
      const_min_other = other.min().count();
    }
    auto const lsb = std::min(get_last_set_bit(), other.get_last_set_bit());
    init_metadata(out, lsb);

    auto const empty_tooth = std::vector<tooth>{};
    auto const empty_saw = saw<SawType>{empty_tooth, traffic_days_};
    auto const loop_it = interleaved_saws<SawType>{
        *this, empty_saw};  // TODO impl single loop iterator
    auto const o_loop_it = interleaved_saws<SawType>{
        other, empty_saw};  // TODO impl single loop iterator
    auto it = loop_it.begin();
    auto o_it = o_loop_it.begin();
    while (it->mam_ + it->travel_dur_.count() >
           o_it->mam_ + o_it.day_offset_ * 24 * 60) {
      --o_it;
    }
    auto last_out_mam_idx = kSawMetadataOffset;
    for (; it != loop_it.end(); ++it) {
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(it->traffic_days_).first;

      for (auto i = lsb; i < get_last_set_bit(); ++i) {
        remaining_traffic_days.set(i + 1, false);
      }

      auto day_offset = 0;
      auto travel_dur_extremum = max ? u16_minutes{0U} : u16_minutes::max();
      // remaining_traffic_days &= ignore_timetable_offset_mask; TODO reuse
      // bitfields?
      // std::cout << "b concat " << std::endl;

      // TODO detect necessary day offset right away and skip loop?
      while (it->mam_ + it->travel_dur_.count() <=
             o_it->mam_ + o_it.day_offset_ * 24 * 60) {
        ++o_it;
      }
      while (true) {
        // std::cout << "a concat " << std::endl;

        --o_it;
        if constexpr (SawType == saw_type::kDay) {
          travel_dur_extremum = std::min(
              u16_minutes{o_it->mam_ - it->mam_ + o_it.day_offset_ * 24 * 60 +
                          o_it->travel_dur_.count()},
              kChMaxEdgeTime);
          break;
        }
        if (day_offset != o_it.day_offset_) {
          if (o_it.day_offset_ > kChMaxWaitingTime / kChDay) {
            break;
          }
          auto const diff =
              static_cast<std::size_t>(o_it.day_offset_ - day_offset);
          for (auto i = 0U;
               i < std::min(diff, static_cast<std::size_t>(lsb + 1U)); ++i) {
            remaining_traffic_days.set(lsb - i, false);
          }
          remaining_traffic_days <<= diff;  // TODO is this correct?
          day_offset = o_it.day_offset_;
        }

        // TODO special case kChMaxEdge time -> remove?
        auto const new_travel_dur = std::min(
            u16_minutes{o_it->mam_ - it->mam_ + o_it.day_offset_ * 24 * 60 +
                        o_it->travel_dur_.count()},
            kChMaxEdgeTime);

        auto conjunction =
            remaining_traffic_days &
            traffic_days_.bitfields_.at(o_it->traffic_days_).first;

        if constexpr (SawType == saw_type::kTrafficDaysPower) {
          if (conjunction.any()) {
            conjunction >>= static_cast<std::size_t>(
                o_it.day_offset_);  // TODO better shift o_it bitfields?
            auto new_tooth = tooth{it->mam_, new_travel_dur, it->traffic_days_};
            auto last_out_mam = saw<SawType>{out, traffic_days_}.begin();
            last_out_mam += last_out_mam_idx - kSawMetadataOffset;

            if (out.size() <= kSawMetadataOffset ||
                non_dominated(new_tooth, last_out_mam, &conjunction, lsb, false,
                              0)) {
              new_tooth.traffic_days_ =
                  traffic_days_.get_or_create(conjunction,
                                              0U);  // TODO avoid recalc

              out.push_back(std::move(new_tooth));
            }
            remaining_traffic_days &=
                ~traffic_days_.bitfields_.at(o_it->traffic_days_).first;
            if (remaining_traffic_days.none()) {
              break;
            }
          }
        } else {
          if (!max) {  // TODO templ
            if (travel_dur_extremum <
                u16_minutes{o_it->mam_ - it->mam_ + o_it.day_offset_ * 24 * 60 +
                            const_min_other}) {
              break;
            }

            if (conjunction.any()) {
              travel_dur_extremum =
                  std::min(travel_dur_extremum, new_travel_dur);
            }
          }
          if (max) {  // TODO templ
            if (conjunction.any()) {
              travel_dur_extremum =
                  std::max(travel_dur_extremum, new_travel_dur);

              remaining_traffic_days &=
                  ~traffic_days_.bitfields_.at(o_it->traffic_days_).first;
              /*std::cout << *it << " " << *o_it << " " << day_offset << " "
                        << lsb << " " << remaining_traffic_days.count() << " "
                        << travel_dur_extremum << std::endl;*/
              if (remaining_traffic_days.none()) {
                break;
              }
              if (travel_dur_extremum >= kChMaxEdgeTime) {
                break;
              }
            }
          }
        }
      }
      if constexpr (SawType == saw_type::kTrafficDaysPower) {
        auto next_it = it;
        ++next_it;
        if (out.size() > kSawMetadataOffset &&
            (next_it == loop_it.end() ||
             next_it->mam_ != out.back().mam_)) {  // TODO ugly
          std::sort(out.begin() + last_out_mam_idx, out.end());
          auto const remaining_it = std::remove_if(
              out.begin() + last_out_mam_idx, out.end(), [&](auto const& e) {
                auto const idx = static_cast<unsigned>(&e - &*out.begin());
                if (idx == last_out_mam_idx) {  // TODO change range +1 instead
                  return false;
                }
                auto& prev = out.at(idx - 1U);
                if (prev.travel_dur_ == e.travel_dur_) {
                  prev.traffic_days_ = traffic_days_.get_or_create(
                      traffic_days_.bitfields_.at(prev.traffic_days_).first |
                          traffic_days_.bitfields_.at(e.traffic_days_).first,
                      std::max(traffic_days_.bitfields_.at(prev.traffic_days_)
                                   .second,
                               traffic_days_.bitfields_.at(e.traffic_days_)
                                   .second));  // TODO cleanup
                  return true;
                }
                return false;
              });
          out.erase(remaining_it, out.end());
          last_out_mam_idx = static_cast<unsigned>(out.size());
        }
      } else {
        if ((max && travel_dur_extremum == u16_minutes{0U}) ||
            (!max && travel_dur_extremum == u16_minutes::max())) {
          continue;
        }
        auto const new_tooth =
            tooth{it->mam_, travel_dur_extremum, it->traffic_days_};
        auto td = traffic_days_.bitfields_.at(it->traffic_days_).first;
        auto oe = saw<SawType>{out, traffic_days_}.end();
        if (out.size() <= kSawMetadataOffset ||
            non_dominated(new_tooth, oe, &td, lsb, false, 0)) {
          out.push_back(std::move(new_tooth));
        }
      }
    }

    if (out.size() <= kSawMetadataOffset) {
      out.clear();  // TODO ugly
    } else {
      auto const out_tmp = out;  // TODO avoid copy
      auto wraparound_saw_begin = saw<SawType>{out_tmp, traffic_days_}.begin();
      auto const remaining_it = std::remove_if(
          out.begin() + kSawMetadataOffset, out.end(), [&](tooth const& e) {
            auto td = traffic_days_.bitfields_.at(e.traffic_days_).first;
            if (non_dominated(e, wraparound_saw_begin, &td, lsb)) {
              return false;
            }
            return true;
          });
      out.erase(remaining_it,
                out.end());  // TODO is this really necessary? power traffic
                             // day subtract? delete markers instead?*/
    }

    auto const s = saw<SawType>{out, traffic_days_};
    return s;
    auto const max_a = this->_max();
    auto const max_b = other._max();

    auto const s_m = s._max();

    if (s_m.first > max_a.first + max_b.first && s_m.first < kChMaxEdgeTime) {
      std::cout << "concat max err " << s_m.first << " " << out[s_m.second]
                << " " << max_a.first << " " << saw_[max_a.second] << " "
                << max_b.first << other.saw_[max_a.second] << std::endl;
      std::cout << s << std::endl;
      std::cout << *this << std::endl;
      std::cout << other << std::endl;
      /*for (auto const& e : s.saw_) {
    for (auto a = this->begin(); a != this->end(); ++a) {
     // if (e.mam_ == )
    }
      }*/
      utl::fail("concat max error");
    }
    auto const mmwt = other.min_max_waiting_time();
    auto const mtd_a = max_travel_dur();
    auto const mtd_b = other.max_travel_dur();
    if (s.min() > max_a.first + max_b.first ||
        s.min() > this->min() + max_b.first ||
        s.min() > max_a.first + mmwt.second + other.min()) {
      std::cout << "concat min err " << s.min() << " " << this->min() << " "
                << other.min() << " " << mtd_a << " " << mtd_b << " "
                << mmwt.first << " " << mmwt.second << " " << max_a.first << " "
                << max_b.first << std::endl;

      std::cout << s << std::endl;
      std::cout << *this << std::endl;
      std::cout << other << std::endl;
      utl::fail("concat min error");
    }

    return s;
  }

  saw<SawType> concat(unsigned const dir,
                      saw<SawType> const& other,
                      bool const max,
                      std::vector<tooth>& out) const {
    if (dir == kReverse) {
      return other.concat(*this, max, out);
    }
    return concat(other, max, out);
  }

  saw<SawType> concat_const(unsigned const dir,
                            saw<saw_type::kConstant> const& other,
                            std::vector<tooth>& out) const {
    utl::verify(out.empty(), "out not empty");
    if (saw_.empty() || other.saw_.empty()) {
      return saw<SawType>{out, traffic_days_};
    }
    auto const d = other.saw_[0].travel_dur_;
    utl::verify(d.count() < 24 * 60, "concat_const more than 24h");
    if (is_constant()) {
      out.push_back({std::numeric_limits<std::int16_t>::max(),
                     std::min(saw_[0].travel_dur_ + d, kChMaxEdgeTime),
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    init_metadata(out, get_last_set_bit());
    for (auto i = kSawMetadataOffset; i < saw_.size(); ++i) {
      auto traffic_days_idx = saw_[i].traffic_days_;
      auto mam = saw_[i].mam_;
      if (dir == kReverse) {
        mam -= d.count();
        if (mam < 0) {
          mam += 24 * 60;
          auto traffic_days =
              traffic_days_.bitfields_.at(traffic_days_idx).first >> 1U;
          traffic_days.set(kTimetableOffset.count() - 1U, false);
          traffic_days_idx = traffic_days_.get_or_create(
              traffic_days,
              traffic_days_.bitfields_.at(traffic_days_idx).second - 1U);
        }
      }
      out.push_back({mam, std::min(saw_[i].travel_dur_ + d, kChMaxEdgeTime),
                     traffic_days_idx});
    }
    return saw<SawType>{out, traffic_days_};
  }

  std::span<tooth const> const saw_;
  traffic_days& traffic_days_;
};

template <saw_type SawType>
struct owning_saw {
  saw<SawType> to_saw(traffic_days& traffic_days) const {
    return saw<SawType>{saw_, traffic_days};
  }
  std::vector<tooth> saw_;
  u16_minutes extremum_;
};

template <saw_type SawType>
struct interleaved_saws {

  struct iterator {
    using difference_type = size_t;
    using value_type = tooth;
    using reference = tooth const&;
    using pointer = tooth const*;
    using iterator_category = std::bidirectional_iterator_tag;

    bool is_a() const {
      if (pos_a_ >= s_.saw_a_.saw_.size()) {
        return false;
      }
      if (pos_b_ >= s_.saw_b_.saw_.size()) {
        return true;
      }
      if (s_.saw_a_.saw_[pos_a_].mam_ == s_.saw_b_.saw_[pos_b_].mam_) {
        return s_.saw_a_.saw_[pos_a_].travel_dur_ <=
               s_.saw_b_.saw_[pos_b_].travel_dur_;
      }
      return s_.saw_a_.saw_[pos_a_].mam_ > s_.saw_b_.saw_[pos_b_].mam_;
    }

    iterator& operator++() {  // TODO take into account travel dur if == mam?
      if (is_a()) {
        ++pos_a_;
      } else {
        ++pos_b_;
      }
      if (pos_a_ >= s_.saw_a_.saw_.size() && pos_b_ >= s_.saw_b_.saw_.size()) {
        --day_offset_;
        pos_a_ = kSawMetadataOffset;
        pos_b_ = kSawMetadataOffset;
      }
      return *this;
    }

    iterator& operator--() {
      if (pos_a_ <= kSawMetadataOffset && pos_b_ <= kSawMetadataOffset) {
        ++day_offset_;
        pos_a_ = std::max(s_.saw_a_.saw_.size(),
                          static_cast<size_t>(kSawMetadataOffset));
        pos_b_ = std::max(s_.saw_b_.saw_.size(),
                          static_cast<size_t>(kSawMetadataOffset));
      }
      if (pos_a_ <= kSawMetadataOffset) {
        --pos_b_;
      } else if (pos_b_ <= kSawMetadataOffset) {
        --pos_a_;
      } else if (s_.saw_a_.saw_[pos_a_ - 1].mam_ ==
                 s_.saw_b_.saw_[pos_b_ - 1].mam_) {
        if (s_.saw_a_.saw_[pos_a_ - 1].travel_dur_ >
            s_.saw_b_.saw_[pos_b_ - 1].travel_dur_) {
          --pos_a_;
        } else {
          --pos_b_;
        }
      } else if (s_.saw_a_.saw_[pos_a_ - 1].mam_ <
                 s_.saw_b_.saw_[pos_b_ - 1].mam_) {
        --pos_a_;
      } else {
        --pos_b_;
      }
      return *this;
    }

    bool operator==(iterator const o) const {
      return s_.saw_a_.begin() == o.s_.saw_a_.begin() &&
             s_.saw_b_.begin() == o.s_.saw_b_.begin() && pos_a_ == o.pos_a_ &&
             pos_b_ == o.pos_b_ && day_offset_ == o.day_offset_;
    }

    bool operator!=(iterator o) const { return !(*this == o); }

    tooth const* operator->() const { return std::addressof(operator*()); }

    tooth const& operator*() const {
      if (is_a()) {
        return s_.saw_a_.saw_[pos_a_];
      }
      return s_.saw_b_.saw_[pos_b_];
    }

    interleaved_saws<SawType> s_;
    size_t pos_a_{kSawMetadataOffset};
    size_t pos_b_{kSawMetadataOffset};
    std::int8_t day_offset_{};
  };

  iterator begin() const {
    utl::verify(saw_a_.valid() && saw_b_.valid(), "invalid saw {} {}",
                saw_a_.saw_.size(), saw_b_.saw_.size());
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_},
                    {kSawMetadataOffset},
                    {kSawMetadataOffset}};
  }

  iterator end() const {
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_},
                    {kSawMetadataOffset},
                    {kSawMetadataOffset},
                    static_cast<std::int8_t>(
                        saw_a_.saw_.empty() && saw_b_.saw_.empty() ? 0 : -1)};
  }

  iterator begin(interleaved_saws<SawType> const& ms) { return ms.begin(); }
  iterator end(interleaved_saws<SawType> const& ms) { return ms.end(); }

  std::reverse_iterator<iterator> rbegin() const {
    return std::make_reverse_iterator(end());
  }

  std::reverse_iterator<iterator> rend() const {
    return std::make_reverse_iterator(begin());
  }

  std::reverse_iterator<iterator> rbegin(interleaved_saws<SawType> const& ms) {
    return ms.rbegin();
  }
  std::reverse_iterator<iterator> rend(interleaved_saws<SawType> const& ms) {
    return ms.rend();
  }

  saw<SawType> const& saw_a_;
  saw<SawType> const& saw_b_;
};

struct ch_label {
  using dist_t = std::uint16_t;
  friend bool operator>(ch_label const& a, ch_label const& b) {
    return a.d_[a.dir_ / kModeOffset] > b.d_[b.dir_ / kModeOffset];
  }
  location_idx_t l_;
  std::array<dist_t, 2> d_;
  std::uint8_t dir_;
};

struct ch_get_bucket {
  ch_label::dist_t operator()(ch_label const& l) const {
    return l.d_[l.dir_ / kModeOffset];
  }
};

struct ch_dist {
  std::array<owning_saw<kChSawType>, 2> d_;
};

}  // namespace nigiri::routing

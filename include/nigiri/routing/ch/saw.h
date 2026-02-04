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
    out.push_back({0, d, bitfield_idx_t::invalid()});
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
        pos_ = 0;
      }
      return *this;
    }

    iterator& operator+=(std::size_t offset) {
      if (!s_.saw_.empty()) {
        auto const tmp = pos_ + offset;
        pos_ += tmp % s_.saw_.size();
        day_offset_ -= tmp / s_.saw_.size();
      }
      return *this;
    }

    iterator& operator--() {
      if (pos_ == 0) {
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
    size_t pos_{};
    std::int8_t day_offset_{};
  };

  iterator begin() const {
    return iterator{saw<SawType>{saw_, traffic_days_}, {}, {}};
  }

  iterator end() const {
    return iterator{saw<SawType>{saw_, traffic_days_},
                    {},
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

  bool less(saw<SawType> const& b, bool const exact_true = false) const {
    if (saw_.empty()) {
      return false;
    }
    if (b.saw_.empty()) {
      return true;
    }
    if constexpr (SawType == saw_type::kConstant) {
      return saw_[0].travel_dur_ < b.saw_[0].travel_dur_;
    }
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
      auto day_offset = 0;
      auto a_it = b_it;
      // std::cout << "b less " << std::endl;
      while (true) {
        // std::cout << "a less " << std::endl;
        --a_it;
        if (a_it.day_offset_ > routing::kMaxTravelTime / 1_days) {
          break;
        }
        if (!a_it.is_a()) {
          continue;  // TODO switch to proper a_it ?
        }
        auto const mam_diff =
            a_it->mam_ - b_it->mam_ + a_it.day_offset_ * 24 * 60;
        auto const remaining_travel_time =
            static_cast<std::int16_t>(b_it->travel_dur_.count()) - mam_diff;

        if constexpr (SawType != saw_type::kDay) {
          if (remaining_travel_time < 0) {  // TODO min const lb
            break;
          }
          if (day_offset != a_it.day_offset_) {
            remaining_traffic_days.set(
                traffic_days_.bitfields_.at(b_it->traffic_days_).second, false);
            remaining_traffic_days <<= 1U;
            day_offset = a_it.day_offset_;
          }
          if (exact_true &&
              remaining_travel_time == a_it->travel_dur_.count() &&
              mam_diff == 0) {  // TODO ugly
            remaining_traffic_days &=
                ~traffic_days_.bitfields_.at(a_it->traffic_days_).first;
            if (remaining_traffic_days.none()) {
              break;
            }
          }
        }

        if (remaining_travel_time < a_it->travel_dur_.count() ||
            (remaining_travel_time == a_it->travel_dur_.count() &&
             mam_diff == 0 && !exact_true)) {
          if constexpr (SawType != saw_type::kDay) {
            if ((remaining_traffic_days &
                 traffic_days_.bitfields_.at(a_it->traffic_days_).first)
                    .none()) {
              continue;
            }
          }
          return false;
        }
        if constexpr (SawType == saw_type::kDay) {
          break;
        }
      }
    }
    return true;
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
    for (auto i = 0U; i < a.saw_.size(); ++i) {
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

  friend std::ostream& operator<<(std::ostream& out, saw<SawType> const& a) {
    for (auto const& e : a.saw_) {
      out << e << " "
          << a.traffic_days_.bitfields_.at(e.traffic_days_).first.count()
          << " ";
      // << a.traffic_days_.bitfields_.at(e.traffic_days_).first << "\n";
    }
    return out;
  }

  u16_minutes max() const {
    if (saw_.empty()) {
      return u16_minutes{kMaxTravelTime.count()};
    }
    if (SawType == saw_type::kConstant) {
      return saw_[0].travel_dur_;
    }
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
          max = std::max(max, a_it->travel_dur_.count() + mam_diff);
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
          max = std::max(max, a_it->travel_dur_.count() + mam_diff);
          remaining_traffic_days &=
              ~traffic_days_.bitfields_.at(a_it->traffic_days_).first;
          if (remaining_traffic_days.none()) {
            break;
          }
        }
      }
    }
    if (max == 0) {
      return u16_minutes{
          kMaxTravelTime
              .count()};  // TODO handle connections that only run on a single
                          // occasion, but multiple times (depending on
                          // proportion of loaded timetable -> kMax?)
    }
    return u16_minutes{max};
  }

  u16_minutes min() const {
    if (saw_.empty()) {
      return u16_minutes{kMaxTravelTime.count()};
    }
    auto min = saw_.front().travel_dur_.count();
    for (auto i = 1U; i < saw_.size(); ++i) {
      min = std::min(min, saw_[i].travel_dur_.count());
    }
    return u16_minutes{min};
  }

  saw<SawType> max(std::vector<tooth>& out,
                   saw_type to = SawType) const {  // TODO insitu
    utl::verify(out.empty(), "out not empty");
    simplify(saw<SawType>{{}, traffic_days_}, out);
    if (to == saw_type::kConstant) {
      auto tmp = saw<SawType>{out, traffic_days_}.max();
      out.clear();
      out.push_back({0, tmp, bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    return saw<SawType>{out, traffic_days_};
  }

  saw<SawType> min(std::vector<tooth>& out,
                   saw_type to = SawType) const {  // TODO insitu
    utl::verify(out.empty(), "out not empty");
    simplify(saw<SawType>{{}, traffic_days_}, out);
    if (to == saw_type::kConstant) {
      auto tmp = saw<SawType>{out, traffic_days_}.min();
      out.clear();
      out.push_back({0, tmp, bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    // TODO ktrafficdays
    return saw<SawType>{out, traffic_days_};
  }

  template <typename Iterator>
  bool non_dominated(tooth const& t,
                     Iterator a_it,
                     bitfield* remaining_traffic_days,
                     bool const not_normalized = false,
                     bool const day_lookahead = routing::kMaxTravelTime /
                                                1_days) const {
    auto const last_set_bit =
        SawType == saw_type::kDay
            ? 0U
            : traffic_days_.bitfields_.at(t.traffic_days_).second;
    auto day_offset = 0;

    while (true) {
      --a_it;
      // std::cout << "nondom " << std::endl;
      if (a_it.day_offset_ > day_lookahead) {
        break;
      }
      auto const mam_diff = a_it->mam_ - t.mam_ + a_it.day_offset_ * 24 * 60;
      auto const remaining_travel_time =
          static_cast<std::int16_t>(t.travel_dur_.count()) - mam_diff;

      if constexpr (SawType == saw_type::kDay) {
        if (remaining_travel_time >= a_it->travel_dur_.count() &&
            a_it->travel_dur_.count() < kChMaxEdgeTime.count()) {
          return false;
        }
        if (!not_normalized) {
          return true;
        }
      }
      if (a_it->travel_dur_.count() >= kChMaxEdgeTime.count()) {
        continue;
      }
      if (remaining_travel_time < 0) {  // TODO min const lb
        break;
      }
      if constexpr (SawType != saw_type::kDay) {
        if (day_offset != a_it.day_offset_) {
          remaining_traffic_days->set(
              last_set_bit, false);  // TODO this eats last_set_bits when
                                     // non_dominated?, move inside dom check
                                     // and set all remaining bits zero there?
          *remaining_traffic_days <<= 1U;
          day_offset = a_it.day_offset_;
        }
        if (remaining_travel_time >= a_it->travel_dur_.count()) {
          *remaining_traffic_days &=
              ~traffic_days_.bitfields_.at(a_it->traffic_days_).first;
          if (remaining_traffic_days->none()) {
            return false;
          }
        }
      }
    }
    if constexpr (SawType == saw_type::kTrafficDaysPower) {
      *remaining_traffic_days >>= static_cast<unsigned>(
          day_offset);  // TODO shift other bitfields instead? – also
      // because this might eat last_set_bits???
    }
    return true;
  }

  saw<SawType> simplify(saw<SawType> const& other,
                        std::vector<tooth>& out) const {
    utl::verify(out.empty(), "out not empty");
    if (saw_.empty() && other.saw_.empty()) {
      return saw<SawType>{out, traffic_days_};
    }
    if constexpr (SawType == saw_type::kConstant) {
      out.push_back(
          {0,
           std::min(saw_.empty() ? u16_minutes::max() : saw_[0].travel_dur_,
                    other.saw_.empty() ? u16_minutes::max()
                                       : other.saw_[0].travel_dur_),
           bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    auto const interleaved = interleaved_saws<SawType>{*this, other};

    for (auto it = interleaved.begin(); it != interleaved.end(); ++it) {
      // std::cout << "simpli " << std::endl;
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(it->traffic_days_).first;
      auto last_set_bit =
          SawType == saw_type::kDay
              ? static_cast<std::uint16_t>(0U)
              : traffic_days_.bitfields_.at(it->traffic_days_).second;
      auto that = it;
      if constexpr (SawType == saw_type::kTrafficDaysPower) {
        ++it;
        while (it != interleaved.end() && that->mam_ == it->mam_ &&
               that->travel_dur_ == it->travel_dur_) {
          remaining_traffic_days |=
              traffic_days_.bitfields_.at(it->traffic_days_).first;
          last_set_bit =
              std::max(last_set_bit,
                       traffic_days_.bitfields_.at(it->traffic_days_).second);
          ++it;
        }
        --it;
      }
      if ((out.empty() &&
           non_dominated(*that, that, &remaining_traffic_days, true)) ||
          (!out.empty() &&
           non_dominated(*that, saw<SawType>{out, traffic_days_}.end(),
                         &remaining_traffic_days, false,
                         0))) {  // TODO wraparound even when non empty?

        auto new_tooth = *that;
        if constexpr (SawType == saw_type::kTrafficDaysPower) {
          new_tooth.traffic_days_ = traffic_days_.get_or_create(
              remaining_traffic_days,
              last_set_bit);  // TODO last set bit update to actual?, dedupl
                              // with concat
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
    if constexpr (SawType == saw_type::kConstant) {
      out.push_back({0,
                     std::min(saw_[0].travel_dur_ + other.saw_[0].travel_dur_,
                              kChMaxEdgeTime),
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    auto const_min_other = 0;
    if (SawType == saw_type::kTrafficDays && !max) {  // TODO constexpr
      const_min_other = other.min().count();
    }
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
    auto last_out_mam_idx = 0U;
    for (; it != loop_it.end(); ++it) {
      auto remaining_traffic_days =
          SawType == saw_type::kDay
              ? bitfield{}
              : traffic_days_.bitfields_.at(it->traffic_days_).first;
      auto const last_set_bit =
          SawType == saw_type::kDay
              ? 0U
              : traffic_days_.bitfields_.at(it->traffic_days_).second;
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
          if (o_it.day_offset_ > routing::kMaxTravelTime / 1_days) {
            break;
          }
          remaining_traffic_days.set(last_set_bit, false);
          remaining_traffic_days <<=
              o_it.day_offset_ - day_offset;  // TODO is this correct?
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
            conjunction >>=
                o_it.day_offset_;  // TODO better shift o_it bitfields?
            auto new_tooth = tooth{it->mam_, new_travel_dur, it->traffic_days_};
            auto last_out_mam = saw<SawType>{out, traffic_days_}.begin();
            last_out_mam += last_out_mam_idx;
            if (out.empty() || non_dominated(new_tooth, last_out_mam,
                                             &conjunction, false, 0)) {
              new_tooth.traffic_days_ = traffic_days_.get_or_create(
                  conjunction,
                  traffic_days_.bitfields_.at(it->traffic_days_).second);
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
        if (next_it == loop_it.end() ||
            next_it->mam_ != out.back().mam_) {  // TODO ugly
          std::sort(out.begin() + last_out_mam_idx, out.end());
          auto const remaining_it = std::remove_if(
              out.begin() + last_out_mam_idx, out.end(), [&](auto const& e) {
                auto const idx = &e - &*out.begin();
                if (idx == 0U) {
                  return false;
                }
                auto& prev = out.at(idx - 1U);
                if (prev.travel_dur_ == e.travel_dur_) {
                  prev.traffic_days_ = traffic_days_.get_or_create(
                      traffic_days_.bitfields_.at(prev.traffic_days_).first |
                          traffic_days_.bitfields_.at(e.traffic_days_).first,
                      std::max(
                          traffic_days_.bitfields_.at(prev.traffic_days_)
                              .second,
                          traffic_days_.bitfields_.at(e.traffic_days_).second));
                  return true;
                }
                return false;
              });
          out.erase(remaining_it, out.end());
          last_out_mam_idx = out.size();
        }
      } else {
        if ((max && travel_dur_extremum == u16_minutes{0U}) ||
            (!max && travel_dur_extremum == u16_minutes::max())) {
          continue;
        }
        auto const new_tooth =
            tooth{it->mam_, travel_dur_extremum, it->traffic_days_};
        auto td = traffic_days_.bitfields_.at(it->traffic_days_).first;
        if (out.empty() ||
            non_dominated(new_tooth, saw<SawType>{out, traffic_days_}.end(),
                          &td, false, 0)) {
          out.push_back(std::move(new_tooth));
        }
      }
    }
    auto const wraparound_saw = saw<SawType>{out, traffic_days_};
    for (auto w = wraparound_saw.begin(); w != wraparound_saw.end(); ++w) {
      auto td = traffic_days_.bitfields_.at(w->traffic_days_).first;
      if (non_dominated(*w, w, &td)) {
        out.erase(out.begin(),
                  out.begin() +
                      static_cast<long>(
                          w.pos_));  // TODO this is not sufficient for
                                     // kTrafficDay, but also not necessary?,
                                     // power traffic day subtract?
        break;
      }
    }
    return saw<SawType>{out, traffic_days_};
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
    if (SawType == saw_type::kConstant) {
      out.push_back({0, std::min(saw_[0].travel_dur_ + d, kChMaxEdgeTime),
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    for (auto i = 0U; i < saw_.size(); ++i) {
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
      if (pos_a_ == s_.saw_a_.saw_.size()) {
        return false;
      }
      if (pos_b_ == s_.saw_b_.saw_.size()) {
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
        pos_a_ = 0;
        pos_b_ = 0;
      }
      return *this;
    }

    iterator& operator--() {
      if (pos_a_ == 0 && pos_b_ == 0) {
        ++day_offset_;
        pos_a_ = s_.saw_a_.saw_.size();
        pos_b_ = s_.saw_b_.saw_.size();
      }
      if (pos_a_ == 0) {
        --pos_b_;
      } else if (pos_b_ == 0) {
        --pos_a_;
      } else if (s_.saw_a_.saw_[pos_a_ - 1].mam_ ==
                 s_.saw_b_.saw_[pos_b_ - 1].mam_) {
        if (s_.saw_a_.saw_[pos_a_].travel_dur_ >
            s_.saw_b_.saw_[pos_b_].travel_dur_) {
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
    size_t pos_a_{};
    size_t pos_b_{};
    std::int8_t day_offset_{};
  };

  iterator begin() const {
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_}};
  }

  iterator end() const {
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_},
                    {},
                    {},
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

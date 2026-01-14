#pragma once

#include <cstdint>
#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {

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
      // auto remaining_traffic_days = traffic_days.at(a.saw_[i].traffic_days_);
      //  remaining_traffic_days &= ignore_timetable_offset_mask; TODO reuse
      //  bitfields?
      //  TODO kTrafficDay
      auto a_it = b_it;
      while (true) {
        --a_it;
        if (a_it.s_.day_offset_ > routing::kMaxTravelTime / 1_days) {
          break;
        }
        if (!a_it.is_a()) {
          continue;  // TODO switch to proper a_it ?
        }
        auto const mam_diff =
            a_it->mam_ - b_it->mam_ + a_it.s_.day_offset_ * 24 * 60;
        auto const remaining_travel_time =
            static_cast<std::int16_t>(b_it->travel_dur_.count()) - mam_diff;

        if (remaining_travel_time < a_it->travel_dur_.count() ||
            (remaining_travel_time == a_it->travel_dur_.count() &&
             mam_diff == 0 &&
             !exact_true)) {  // TODO exactly identical should not be a < b ?
          return false;
        }
        break;
        // TODO traffic days
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

  friend std::ostream& operator<<(std::ostream& out, saw<SawType> const& a) {
    for (auto const& e : a.saw_) {
      out << "(" << e.mam_ << "," << e.travel_dur_ << ")";
    }
    return out;
  }

  u16_minutes max() const {
    // TODO ktrafficdays / simplify?
    if (saw_.empty()) {
      return u16_minutes::max();
    }
    if (SawType == saw_type::kConstant) {
      return saw_[0].travel_dur_;
    }
    auto max = saw_.back().mam_ + 24 * 60 - saw_.front().mam_ +
               saw_.back().travel_dur_.count();
    for (auto i = 1U; i < saw_.size(); ++i) {
      max = std::max(max, saw_[i - 1].mam_ - saw_[i].mam_ +
                              saw_[i - 1].travel_dur_.count());
    }
    return u16_minutes{max};
  }

  u16_minutes min() const {
    if (saw_.empty()) {
      return u16_minutes::max();
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
    // TODO ktrafficdays
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
  bool non_dominated(tooth const& t, Iterator begin, Iterator end) const {
    auto remaining_traffic_days = SawType == saw_type::kDay
                                      ? bitfield{}
                                      : traffic_days_.at(t.traffic_days_);

    for (auto rev = begin; rev != end; ++rev) {
      if constexpr (SawType != saw_type::kDay) {
        remaining_traffic_days &= ~traffic_days_.at(rev->traffic_days_);
      }
      auto const remaining_travel_time =
          static_cast<std::int32_t>(t.travel_dur_.count()) - rev->mam_ + t.mam_;
      if constexpr (SawType == saw_type::kDay) {
        return remaining_travel_time < rev->travel_dur_.count();
      }
      if (remaining_travel_time < 0) {
        return true;  // TOOO no need to always look till the end for
                      // kTrafficDays?
      }
      if (remaining_traffic_days.none() &&
          remaining_travel_time >= rev->travel_dur_.count()) {
        return false;
      }
    }
    return true;
  }

  tooth last_duration_of_day(interleaved_saws<SawType> interleaved) const {
    auto const& last = *interleaved.begin();
    auto last_duration_of_day = last.travel_dur_;
    for (auto it = interleaved.rbegin(); it != interleaved.rend(); ++it) {
      if (last.mam_ + static_cast<std::int32_t>(last.travel_dur_.count()) -
              24 * 60 <
          it->mam_) {
        break;
      }
      last_duration_of_day =
          std::min(last_duration_of_day,
                   u16_minutes{it->travel_dur_.count() + 24 * 60 + it->mam_ -
                               last.mam_});  // TODO traffic days
    }
    return {last.mam_, last_duration_of_day,
            bitfield_idx_t::invalid()};  // TODO traffic days
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
    out.push_back(last_duration_of_day(interleaved));

    for (auto it = ++interleaved.begin(); it != interleaved.end(); ++it) {
      if (non_dominated(*it, out.rbegin(), out.rend())) {
        out.push_back(*it);
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
      out.push_back({0, saw_[0].travel_dur_ + other.saw_[0].travel_dur_,
                     bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    auto const empty_tooth = std::vector<tooth>{};
    auto const empty_saw = saw<kChSawType>{empty_tooth, traffic_days_};
    auto const loop_it = interleaved_saws<SawType>{
        *this, empty_saw};  // TODO impl single loop iterator
    auto const o_loop_it = interleaved_saws<SawType>{
        other, empty_saw};  // TODO impl single loop iterator
    auto it = loop_it.begin();
    auto o_it = o_loop_it.begin();
    while (it->mam_ + it->travel_dur_.count() >
           o_it->mam_ + o_it.s_.day_offset_ * 24 * 60) {
      --o_it;
    }
    for (; it != loop_it.end(); ++it) {
      auto remaining_traffic_days = SawType == saw_type::kDay
                                        ? bitfield{}
                                        : traffic_days_.at(it->traffic_days_);
      // remaining_traffic_days &= ignore_timetable_offset_mask; TODO reuse
      // bitfields?
      while (true) {
        if (it->mam_ + it->travel_dur_.count() <=
            o_it->mam_ + o_it.s_.day_offset_ * 24 * 60) {
          ++o_it;
          continue;
        }
        --o_it;
        if (SawType == saw_type::kDay) {
          break;
        }
        if (!max &&
            (remaining_traffic_days & traffic_days_.at(o_it->traffic_days_))
                .any()) {
          break;
        }
        remaining_traffic_days &= ~traffic_days_.at(o_it->traffic_days_);
        if (max && remaining_traffic_days.none()) {
          break;
        }
        if (o_it.s_.day_offset_ > routing::kMaxTravelTime / 1_days) {
          break;
        }
        /*if (SawType != saw_type::kDay) {
          remaining_traffic_days <<= 1U; // TODO day offset
        }*/
      }
      auto const new_tooth = tooth{
          it->mam_,
          u16_minutes{o_it->mam_ - it->mam_ + o_it.s_.day_offset_ * 24 * 60 +
                      o_it->travel_dur_.count()},
          it->traffic_days_};
      if (non_dominated(new_tooth, out.rbegin(), out.rend())) {
        out.push_back(new_tooth);  // TODO wraparound
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
    if (SawType == saw_type::kConstant) {
      out.push_back({0, saw_[0].travel_dur_ + d, bitfield_idx_t::invalid()});
      return saw<SawType>{out, traffic_days_};
    }
    for (auto i = 0U; i < saw_.size(); ++i) {
      auto mam = saw_[i].mam_;
      if (dir == kReverse) {
        mam -= d.count();
        if (mam < 0) {
          mam += 24 * 60;  // TODO traffic days
        }
      }
      out.push_back({mam, saw_[i].travel_dur_ + d, bitfield_idx_t::invalid()});
    }
    return saw<SawType>{out, traffic_days_};
  }

  std::span<tooth const> const saw_;
  vector_map<bitfield_idx_t, bitfield> const& traffic_days_;
};

template <saw_type SawType>
struct owning_saw {
  saw<SawType> to_saw(
      vector_map<bitfield_idx_t, bitfield> const& traffic_days) const {
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

    iterator& operator++() {
      if (s_.pos_a_ == s_.saw_a_.saw_.size()) {
        ++s_.pos_b_;
      } else if (s_.pos_b_ == s_.saw_b_.saw_.size()) {
        ++s_.pos_a_;
      } else if (s_.saw_a_.saw_[s_.pos_a_].mam_ >=
                 s_.saw_b_.saw_[s_.pos_b_].mam_) {
        ++s_.pos_a_;
      } else {
        ++s_.pos_b_;
      }
      if (s_.pos_a_ >= s_.saw_a_.saw_.size() &&
          s_.pos_b_ >= s_.saw_b_.saw_.size()) {
        --s_.day_offset_;
        s_.pos_a_ = 0;
        s_.pos_b_ = 0;
      }
      return *this;
    }

    iterator& operator--() {
      if (s_.pos_a_ == 0 && s_.pos_b_ == 0) {
        ++s_.day_offset_;
        s_.pos_a_ = s_.saw_a_.saw_.size();
        s_.pos_b_ = s_.saw_b_.saw_.size();
      }
      if (s_.pos_a_ == 0) {
        --s_.pos_b_;
      } else if (s_.pos_b_ == 0) {
        --s_.pos_a_;
      } else if (s_.saw_a_.saw_[s_.pos_a_ - 1].mam_ <
                 s_.saw_b_.saw_[s_.pos_b_ - 1].mam_) {
        --s_.pos_a_;
      } else {
        --s_.pos_b_;
      }
      return *this;
    }

    bool operator==(iterator const o) const {
      return &s_.saw_a_ == &o.s_.saw_a_ && &s_.saw_b_ == &o.s_.saw_b_ &&
             s_.pos_a_ == o.s_.pos_a_ && s_.pos_b_ == o.s_.pos_b_ &&
             s_.day_offset_ == o.s_.day_offset_;
    }

    bool operator!=(iterator o) const { return !(*this == o); }

    tooth const* operator->() const { return std::addressof(operator*()); }

    bool is_a() const {
      if (s_.pos_a_ == s_.saw_a_.saw_.size()) {
        return false;
      }
      if (s_.pos_b_ == s_.saw_b_.saw_.size()) {
        return true;
      }
      return s_.saw_a_.saw_[s_.pos_a_].mam_ >= s_.saw_b_.saw_[s_.pos_b_].mam_;
    }

    tooth const& operator*() const {
      if (is_a()) {
        return s_.saw_a_.saw_[s_.pos_a_];
      }
      return s_.saw_b_.saw_[s_.pos_b_];
    }

    interleaved_saws<SawType> s_;
  };

  iterator begin() const {
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_}};
  }

  iterator end() const {
    return iterator{interleaved_saws<SawType>{saw_a_, saw_b_, {}, {}, -1}};
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
  size_t pos_a_{};
  size_t pos_b_{};
  std::int8_t day_offset_{};
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

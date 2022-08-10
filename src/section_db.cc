#include "nigiri/section_db.h"

#include "date/date.h"

namespace nigiri {

std::ostream& operator<<(std::ostream& out, tz_offsets const& offsets) {
  out << "(general_offset=" << offsets.offset_ << ", season=(";
  if (offsets.season_.has_value()) {
    out << "offset=" << offsets.season_->offset_ << ", ";

    out << "begin=";
    date::to_stream(out, "%F", offsets.season_->begin_);
    out << " " << offsets.season_->season_begin_mam_;

    out << ", end=";
    date::to_stream(out, "%F", offsets.season_->end_);
    out << " " << offsets.season_->season_end_mam_;
  } else {
    out << "none";
  }
  out << "))";
  return out;
}

}  // namespace nigiri
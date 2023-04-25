#include "nigiri/timetable.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/overloaded.h"

#include "nigiri/print_transport.h"

namespace nigiri {

constexpr auto const kMode =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_STATIC_VERSION;

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  auto const internal = tt.internal_interval_days();
  auto const num_days =
      static_cast<size_t>((internal.to_ - internal.from_ + 1_days) / 1_days);
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << reverse(traffic_days.to_string().substr(kMaxDays - num_days))
        << "\n";
    for (auto d = internal.from_; d != internal.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - internal.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, {transport_idx, day_idx});
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

cista::wrapped<timetable> timetable::read(cista::memory_holder&& mem) {
  return std::visit(
      utl::overloaded{
          [&](cista::buf<cista::mmap>& b) {
            auto const ptr =
                reinterpret_cast<timetable*>(&b[cista::data_start(kMode)]);
            return cista::wrapped{std::move(mem), ptr};
          },
          [&](cista::buffer& b) {
            auto const ptr = cista::deserialize<timetable, kMode>(b);
            return cista::wrapped{std::move(mem), ptr};
          },
          [&](cista::byte_buf& b) {
            auto const ptr = cista::deserialize<timetable, kMode>(b);
            return cista::wrapped{std::move(mem), ptr};
          }},
      mem);
}

void timetable::write(std::filesystem::path const& p) const {
  auto mmap = cista::mmap{p.string().c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));

  {
    auto const timer = scoped_timer{"timetable.write"};
    cista::serialize<kMode>(writer, *this);
  }
}

void timetable::write(cista::memory_holder& mem) const {
  std::visit(utl::overloaded{[&](cista::buf<cista::mmap>& writer) {
                               cista::serialize<kMode>(writer, *this);
                             },
                             [&](cista::buffer&) {
                               throw std::runtime_error{"not supported"};
                             },
                             [&](cista::byte_buf& b) {
                               auto writer = cista::buf{std::move(b)};
                               cista::serialize<kMode>(writer, *this);
                               b = std::move(writer.buf_);
                             }},
             mem);
}

}  // namespace nigiri

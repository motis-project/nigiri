#include <memory>

#include "rust/cxx.h"

#include "nigiri/timetable.h"

namespace nigiri::rust {

using Timetable = cista::wrapped<nigiri::timetable>;

struct LoaderConfig;

std::unique_ptr<Timetable> parse_timetables(
    ::rust::Vec<::rust::String> const& paths,
    LoaderConfig const&,
    ::rust::Str start_date,
    std::uint32_t num_days);

void dump_timetable(Timetable const&, ::rust::Str path);

std::unique_ptr<Timetable> load_timetable(::rust::Str path);

}  // namespace nigiri::rust
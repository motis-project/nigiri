#include "nigiri/common/gunzip.h"

#include <sstream>

#include "boost/iostreams/copy.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "boost/iostreams/filtering_streambuf.hpp"

namespace nigiri {

std::string gunzip(std::string_view s) {
  auto const src = boost::iostreams::array_source{s.data(), s.size()};
  auto is = boost::iostreams::filtering_istream{};
  auto os = std::stringstream{};
  is.push(boost::iostreams::gzip_decompressor{});
  is.push(src);
  boost::iostreams::copy(is, os);
  return os.str();
}

}  // namespace nigiri
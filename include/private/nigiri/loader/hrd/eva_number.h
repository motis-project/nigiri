#pragma once

#include "cista/strong.h"

#include "utl/parser/arg_parser.h"

namespace nigiri::loader::hrd {

using eva_number = cista::strong<unsigned, struct _eva_num>;

inline eva_number parse_eva_number(utl::cstr s) {
  return eva_number{utl::parse_verify<unsigned>(s)};
}

}  // namespace nigiri::loader::hrd

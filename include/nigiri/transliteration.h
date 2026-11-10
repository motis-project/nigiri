#include <optional>
#include <string>
#include <vector>

#include "unicode/uscript.h"

#include "nigiri/types.h"

namespace nigiri {

std::vector<UScriptCode> lang_get_scripts(char const* lang_code);
std::string lang_code_without_country(std::string const& locale);
std::string find_best_transliterator(std::string_view source_lang);
std::optional<translation> transliterate(std::string_view source_string,
                                         std::string_view source_lang);

}  // namespace nigiri
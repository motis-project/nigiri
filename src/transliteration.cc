#include "nigiri/transliteration.h"

#include <string>
#include <vector>

#include "utl/verify.h"

#include "unicode/translit.h"
#include "unicode/uloc.h"
#include "unicode/uscript.h"

#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri {

/// lang_code needs to be null-terminated
std::vector<UScriptCode> lang_get_scripts(char const* lang_code) {
  UErrorCode status = U_ZERO_ERROR;
  char locale[ULOC_FULLNAME_CAPACITY];

  auto const locale_len =
      uloc_addLikelySubtags(lang_code, locale, sizeof(locale), &status);

  if (U_FAILURE(status)) {
    log(log_lvl::error, "timetable.lang_get_scripts",
        "icu error when adding subtags: {}", u_errorName(status));
    return {};
  }

  if (static_cast<std::size_t>(locale_len) >= sizeof(locale)) {
    return {};
  }

  char script[USCRIPT_CODE_LIMIT];
  status = U_ZERO_ERROR;

  auto const script_len =
      uloc_getScript(locale, script, sizeof(script), &status);

  if (U_FAILURE(status)) {
    log(log_lvl::error, "timetable.lang_get_scripts",
        "icu error when detecting script: {}", u_errorName(status));
    return {};
  }

  if (script_len == 0 ||
      static_cast<std::size_t>(script_len) >= sizeof(script)) {
    return {};
  }

  UScriptCode codes[10];
  status = U_ZERO_ERROR;

  auto const count = uscript_getCode(script, codes, 10, &status);

  if (U_FAILURE(status)) {
    log(log_lvl::error, "timetable.lang_get_scripts",
        "icu error when getting script code: {}", u_errorName(status));
    return {};
  }

  if (count <= 0) {
    return {};
  }

  return {codes, codes + count};
}

std::string lang_code_without_country(std::string const& locale) {
  UErrorCode status = U_ZERO_ERROR;
  char language[ULOC_LANG_CAPACITY];
  char script[ULOC_SCRIPT_CAPACITY];

  uloc_getLanguage(locale.c_str(), language, sizeof(language), &status);
  if (U_FAILURE(status)) {
    return locale;
  }

  status = U_ZERO_ERROR;
  uloc_getScript(locale.c_str(), script, sizeof(script), &status);
  if (U_FAILURE(status)) {
    return locale;
  }

  if (script[0] != '\0') {
    return std::string{language} + "-" + script;
  }

  return language;
}

std::string find_best_transliterator(std::string_view source_lang) {
  UErrorCode ec = U_ZERO_ERROR;

  auto const ids = std::unique_ptr<icu::StringEnumeration>(
      icu::Transliterator::getAvailableIDs(ec));
  utl::verify(!U_FAILURE(ec), "icu build broken");

  auto const prefix = std::format("{}-{}_Latn", source_lang, source_lang);

  while (auto const* id = ids->snext(ec)) {
    std::string candidate;
    id->toUTF8String(candidate);

    if (candidate.starts_with(prefix)) {
      return candidate;
    }
  }

  return std::format("{}-Latn", source_lang);
}

std::optional<translation> transliterate(std::string_view source_string,
                                         std::string_view source_lang) {

  auto sl = std::string{source_lang};
  if (std::ranges::contains(lang_get_scripts(sl.c_str()),
                            UScriptCode::USCRIPT_LATIN)) {
    return std::nullopt;
  }

  auto const translit_id = find_best_transliterator(source_lang);

  UErrorCode ec = U_ZERO_ERROR;
  auto const transliterator =
      std::unique_ptr<icu::Transliterator>(icu::Transliterator::createInstance(
          icu::UnicodeString::fromUTF8(translit_id),
          UTransDirection::UTRANS_FORWARD, ec));
  if (!transliterator) {
    log(log_lvl::debug, "timetable.transliterate",
        "No transliterator for {} :(", translit_id);
    return std::nullopt;
  }
  auto replaceable = icu::UnicodeString::fromUTF8(source_string);
  transliterator->transliterate(replaceable);
  auto const transliterated = replaceable.toUTF8String<std::string>();

  if (transliterated == source_string) {
    return std::nullopt;
  }

  auto lang = generic_string{};
  lang.set_owning(std::format("{}-Latn", source_lang));

  auto text = generic_string{};
  text.set_owning(transliterated);

  auto t = translation{};
  t.language_ = lang;
  t.text_ = text;

  return std::optional{t};
}

}  // namespace nigiri
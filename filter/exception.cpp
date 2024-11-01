#include "exception.h"

#include <sstream>

namespace {

[[nodiscard]] auto FormatMessage(const std::string& what, const std::source_location& loc) -> std::string {
  std::ostringstream stream;
  stream << loc.file_name() << ':' << loc.line() << ": " << what;
  return stream.str();
}

}  // namespace

Exception::Exception(const std::string& what, const std::source_location& loc)
    : std::runtime_error(FormatMessage(what, loc)) {}

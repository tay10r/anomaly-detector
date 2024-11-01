#pragma once

#include <source_location>
#include <stdexcept>

class Exception : public std::runtime_error {
 public:
  Exception(const std::string& what, const std::source_location& location = std::source_location::current());
};

#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "ember/check.h"
#include "ember/status.h"

namespace ember {

// Either a value or the reason there is no value. The engine does not use exceptions, so anything
// that builds a value and can fail on bad input returns one of these.
//
// Reading the value of a failed StatusOr is a bug and aborts, so a forgotten ok() check cannot turn
// into reading garbage.
template <typename T>
class [[nodiscard]] StatusOr {
  static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>, "use Status directly");

 public:
  // Both are implicit on purpose, so a function can simply `return value;` or `return status;`.
  StatusOr(T value) : state_(std::in_place_index<1>, std::move(value)) {}
  StatusOr(Status status) : state_(std::in_place_index<0>, std::move(status)) {
    EMBER_CHECK(!std::get<0>(state_).ok(), "a StatusOr built from a Status must hold an error");
  }

  bool ok() const { return state_.index() == 1; }
  Status status() const { return ok() ? Status::Ok() : std::get<0>(state_); }

  const T& value() const& {
    EMBER_CHECK(ok(), "read the value of a failed StatusOr");
    return std::get<1>(state_);
  }
  T& value() & {
    EMBER_CHECK(ok(), "read the value of a failed StatusOr");
    return std::get<1>(state_);
  }
  // By value on purpose. Returning a reference into a dying temporary would make
  // `const auto& shape = *Shape::Make(...)` compile and then dangle.
  T value() && {
    EMBER_CHECK(ok(), "read the value of a failed StatusOr");
    return std::get<1>(std::move(state_));
  }

  const T& operator*() const& { return value(); }
  T& operator*() & { return value(); }
  T operator*() && { return std::move(*this).value(); }
  const T* operator->() const { return &value(); }
  T* operator->() { return &value(); }

 private:
  std::variant<Status, T> state_;
};

}  // namespace ember

#define EMBER_INTERNAL_CONCAT_(a, b) a##b
#define EMBER_INTERNAL_CONCAT(a, b) EMBER_INTERNAL_CONCAT_(a, b)

// Returns from the current function if the expression gives a failed Status.
#define EMBER_RETURN_IF_ERROR(expression)                                    \
  do {                                                                       \
    if (::ember::Status ember_status_ = (expression); !ember_status_.ok()) { \
      return ember_status_;                                                  \
    }                                                                        \
  } while (false)

#define EMBER_INTERNAL_ASSIGN_OR_RETURN(temporary, target, expression) \
  auto temporary = (expression);                                       \
  if (!temporary.ok()) return temporary.status();                      \
  target = std::move(temporary).value()

// `EMBER_ASSIGN_OR_RETURN(const Shape shape, Shape::Make(dims));` declares and fills `shape`, or
// returns the error from the current function. It expands to several statements, so use it as a
// statement of its own, in braces if it follows an `if`, and at most once per line.
#define EMBER_ASSIGN_OR_RETURN(target, expression)                                           \
  EMBER_INTERNAL_ASSIGN_OR_RETURN(EMBER_INTERNAL_CONCAT(ember_status_or_, __LINE__), target, \
                                  expression)

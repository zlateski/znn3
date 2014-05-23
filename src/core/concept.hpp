#pragma once

#include <type_traits>

#define CONCEPT_IS_CALLABLE(__name__)                                   \
                                                                        \
    template <typename C, typename F, typename = void>                  \
    struct is_callable_ ## __name__ : std::false_type {};               \
                                                                        \
    template <typename C, typename R, typename... A>                    \
    struct is_callable_ ## __name__ <C, R(A...)                         \
                                     , typename std::enable_if          \
                                     < std::is_same<R, void>::value ||  \
                                       std::is_convertible              \
                                       < decltype(std::declval<C>().    \
                                                  __name__              \
                                                  (std::declval<A>()...) \
                                             ), R>::value               \
                                       >::type                          \
                                     > : public std::true_type {}

#ifndef GEMM_HPP
#define GEMM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 20
# define NJ 25
# define NK 30
#elif defined(SMALL_DATASET)
# define NI 60
# define NJ 70
# define NK 80
#elif defined(MEDIUM_DATASET)
# define NI 200
# define NJ 220
# define NK 240
#elif defined(LARGE_DATASET)
# define NI 1000
# define NJ 1100
# define NK 1200
#elif defined(EXTRALARGE_DATASET)
# define NI 2000
# define NJ 2300
# define NK 2600
#endif

#endif // GEMM_HPP

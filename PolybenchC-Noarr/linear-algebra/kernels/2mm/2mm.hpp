#ifndef _2MM_HPP
#define _2MM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 16
# define NJ 18
# define NK 22
# define NL 24
#elif defined(SMALL_DATASET)
# define NI 40
# define NJ 50
# define NK 70
# define NL 80
#elif defined(MEDIUM_DATASET)
# define NI 180
# define NJ 190
# define NK 210
# define NL 220
#elif defined(LARGE_DATASET)
# define NI 800
# define NJ 900
# define NK 1100
# define NL 1200
#elif defined(EXTRALARGE_DATASET)
# define NI 1600
# define NJ 1800
# define NK 2200
# define NL 2400
#endif

#endif // _2MM_HPP

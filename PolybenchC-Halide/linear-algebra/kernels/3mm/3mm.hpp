#ifndef _3MM_HPP
#define _3MM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NI 16
# define NJ 18
# define NK 20
# define NL 22
# define NM 24
#elif defined(SMALL_DATASET)
# define NI 40
# define NJ 50
# define NK 60
# define NL 70
# define NM 80
#elif defined(MEDIUM_DATASET)
# define NI 180
# define NJ 190
# define NK 200
# define NL 210
# define NM 220
#elif defined(LARGE_DATASET)
# define NI 800
# define NJ 900
# define NK 1000
# define NL 1100
# define NM 1200
#elif defined(EXTRALARGE_DATASET)
# define NI 1600
# define NJ 1800
# define NK 2000
# define NL 2200
# define NM 2400
#endif

#endif // _3MM_HPP

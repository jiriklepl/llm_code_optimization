#ifndef DOITGEN_HPP
#define DOITGEN_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define NR 10
# define NQ 8
# define NP 12
#elif defined(SMALL_DATASET)
# define NR 25
# define NQ 20
# define NP 30
#elif defined(MEDIUM_DATASET)
# define NR 50
# define NQ 40
# define NP 60
#elif defined(LARGE_DATASET)
# define NR 150
# define NQ 140
# define NP 160
#elif defined(EXTRALARGE_DATASET)
# define NR 250
# define NQ 220
# define NP 270
#endif

#endif // DOITGEN_HPP

#ifndef FDTD_2D_HPP
#define FDTD_2D_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define TMAX 20
# define NX 20
# define NY 30
#elif defined(SMALL_DATASET)
# define TMAX 40
# define NX 60
# define NY 80
#elif defined(MEDIUM_DATASET)
# define TMAX 100
# define NX 200
# define NY 240
#elif defined(LARGE_DATASET)
# define TMAX 500
# define NX 1000
# define NY 1200
#elif defined(EXTRALARGE_DATASET)
# define TMAX 1000
# define NX 2000
# define NY 2600
#endif

#endif // FDTD_2D_HPP

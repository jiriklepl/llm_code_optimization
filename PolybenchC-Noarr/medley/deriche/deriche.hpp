#ifndef DERICHE_HPP
#define DERICHE_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define W 64
# define H 64
#elif defined(SMALL_DATASET)
# define W 192
# define H 128
#elif defined(MEDIUM_DATASET)
# define W 720
# define H 480
#elif defined(LARGE_DATASET)
# define W 4096
# define H 2160
#elif defined(EXTRALARGE_DATASET)
# define W 7680
# define H 4320
#endif

#endif // DERICHE_HPP

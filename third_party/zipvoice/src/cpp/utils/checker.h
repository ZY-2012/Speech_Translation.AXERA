/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Utility: checker macros
 *
 * Adapted from melotts.axera-main
 **************************************************************************************************/

#pragma once

#include "utils/logger.h"

#define CHECK_PTR(p)                     \
    do {                                 \
        if (!p) {                        \
            ALOGE("%s nil pointer\n", #p); \
            return -1;                   \
        }                                \
    } while (0)

#define CHECK_INITED(p)                     \
    do {                                    \
        if (!p->HasInit()) {                \
            ALOGE("%s has not init\n", #p); \
            return -1;                      \
        }                                   \
    } while (0)

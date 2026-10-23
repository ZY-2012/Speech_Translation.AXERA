/**************************************************************************************************
 * ZipVoice C++ — Unified EngineWrapper implementation (AXERA & AXCL)
 *
 * Conditionally includes the platform-specific implementation.
 **************************************************************************************************/

#if defined(AX650) || defined(AX630C) || defined(AX620Q)
    #include "EngineWrapper_axera.cpp"
#elif defined(AXCL)
    #include "EngineWrapper_axcl.cpp"
#else
    #error "Unknown platform. Define AX650, AX630C, AX620Q, or AXCL."
#endif

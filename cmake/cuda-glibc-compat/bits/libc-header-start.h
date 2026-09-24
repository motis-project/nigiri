/* CUDA 12.x declares rsqrt, rsqrtf, sinpi, sinpif, cospi and cospif with no
   exception specification. glibc 2.43 declares the same six as C23 functions
   with noexcept(true), and a redeclaration with a mismatched exception
   specification is a hard error, so every .cu including <cmath> fails to
   compile - CMake's own CUDA compiler-identification test included.
   12.9 is the last CUDA 12.x release and 13.x dropped Pascal, so a Pascal
   build on a current glibc has no toolkit-side fix available.
   glibc gates those declarations on __GLIBC_USE (IEC_60559_FUNCS_EXT_C23),
   set in bits/libc-header-start.h. That header deliberately has no include
   guard so it can be re-read as feature macros change, so re-include it and
   then turn the C23 math additions back off - the setting then holds for every
   re-read in the translation unit.
   Gated on the nvcc version here rather than in CMake because the flag has to
   be in place before enable_language(CUDA), which is too early to probe. Only
   on the include path for CUDA compilation, so host builds never see it. */

#include_next <bits/libc-header-start.h>

/* CUDA 13.0 declares these compatibly; leave a fixed toolkit alone. */
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 13
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#endif

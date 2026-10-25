#pragma once

// A plain C interface to World, compiled into outerworld_py.dll and loaded from Python with ctypes.
// C is used (not C++) because the DLL is built with MinGW while Python is built with MSVC:
// the two compilers' C++ is incompatible, but their C calling convention is the same.

#ifdef __cplusplus
extern "C" {
#endif

#define OW_API __declspec(dllexport)

typedef struct OwWorld OwWorld;   // opaque: Python only ever holds a pointer

OW_API int ow_obs_size(void);
OW_API int ow_num_actions(void);

OW_API OwWorld* ow_create(void);              // starts on the test level
OW_API void ow_destroy(OwWorld* world);

OW_API void ow_load_test_level(OwWorld* world);
OW_API void ow_reset(OwWorld* world);

// Advances one fixed timestep. action is 0..ow_num_actions()-1.
// Returns 0 = running, 1 = crashed, 2 = reached goal.
OW_API int ow_step(OwWorld* world, int action);

OW_API void ow_observe(const OwWorld* world, float* out);   // writes ow_obs_size() floats
OW_API float ow_goal_distance(const OwWorld* world);
OW_API int ow_step_count(const OwWorld* world);

#ifdef __cplusplus
}
#endif

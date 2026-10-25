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
// A random level from the generator. difficulty: 0 = easy .. 1 = hard.
OW_API void ow_load_generated(OwWorld* world, unsigned int seed, float difficulty);
OW_API void ow_reset(OwWorld* world);

// Advances one fixed timestep. action is 0..ow_num_actions()-1.
// Returns 0 = running, 1 = crashed, 2 = reached goal.
OW_API int ow_step(OwWorld* world, int action);

OW_API void ow_observe(const OwWorld* world, float* out);   // writes ow_obs_size() floats
OW_API float ow_goal_distance(const OwWorld* world);   // straight line to the goal
OW_API float ow_path_distance(const OwWorld* world);   // travel distance around obstacles
OW_API int ow_step_count(const OwWorld* world);

// ---- Batched: many worlds stepped by one call, so Python pays its per-call overhead once per
// batch instead of once per world. Every array argument holds one entry per world, in order.

typedef struct OwBatch OwBatch;

OW_API OwBatch* ow_batch_create(int num_worlds);   // all start on the test level
OW_API void ow_batch_destroy(OwBatch* batch);

OW_API void ow_batch_load_test_level(OwBatch* batch, int index);
OW_API void ow_batch_load_generated(OwBatch* batch, int index, unsigned int seed, float difficulty);

// Advances every world one fixed timestep. shaping chooses which distance is written:
// 0 = straight line, 1 = path around obstacles, 2 = none (writes 0).
// Writes results (0 running, 1 crashed, 2 goal), step counts, distances, and
// num_worlds * ow_obs_size() observations, all taken after the step.
OW_API void ow_batch_step(OwBatch* batch, const int* actions, int shaping,
                          int* results, int* step_counts, float* distances, float* observations);

// One world's observation and distance, e.g. right after loading it a new level.
OW_API void ow_batch_observe_one(const OwBatch* batch, int index, float* out);
OW_API float ow_batch_distance_one(const OwBatch* batch, int index, int shaping);

#ifdef __cplusplus
}
#endif

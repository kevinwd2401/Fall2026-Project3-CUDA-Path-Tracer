#pragma once

// Compile-time benchmark controls

#ifndef ENABLE_TONEMAPPING
// 1 applies the ACES-fitted curve and display gamma; 0 outputs clamped
// scene-linear radiance for an un-tonemapped comparison.
#define ENABLE_TONEMAPPING 1
#endif

#ifndef USE_BVH
// 1 uses the flattened SAH BVH; 0 skips BVH construction/upload and tests
// every primitive in the linear traversal kernels for a baseline.
#define USE_BVH 1
#endif

#ifndef MATERIAL_SORT
#define MATERIAL_SORT 0
#endif

#ifndef MORTON_SORT
#define MORTON_SORT 1
#endif

#ifndef MORTON_SORT_MIN_RAYS
#define MORTON_SORT_MIN_RAYS 4096
#endif

#ifndef TRIANGLE_TRAVERSAL
#define TRIANGLE_TRAVERSAL 1
#endif

#ifndef RUSSIAN_ROULETTE
#define RUSSIAN_ROULETTE 1
#endif

#ifndef RUSSIAN_ROULETTE_START_DEPTH
#define RUSSIAN_ROULETTE_START_DEPTH 3
#endif

#ifndef DEPTH_OF_FIELD
#define DEPTH_OF_FIELD 1
#endif

#ifndef ERRORCHECK
#define ERRORCHECK 0
#endif

// CPU BVH construction.  Leaf size changes the number of primitive tests per
// leaf versus the number of AABB tests.  More SAH bins cost more build time but
// can produce better splits for large meshes.
#ifndef BVH_LEAF_SIZE
#define BVH_LEAF_SIZE 4
#endif

#ifndef BVH_SAH_BINS
#define BVH_SAH_BINS 16
#endif

// Keep this enabled for correct alpha-masked/blended direct lighting.  Set to
// 0 only for an opaque-shadow performance baseline; it intentionally treats
// every shadow hit as opaque even when the scene contains alpha materials.
#ifndef SHADOW_ALPHA_TRANSMISSION
#define SHADOW_ALPHA_TRANSMISSION 1
#endif

// Shadow compaction removes inactive direct-light rays before traversal, but
// requires a device-to-host count readback.  Set to 0 to benchmark the opposite
// tradeoff: launch the full path queue and skip inactive shadow slots in-kernel.
#ifndef SHADOW_RAY_COMPACTION
#define SHADOW_RAY_COMPACTION 1
#endif

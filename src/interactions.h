#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materials (such as dielectric).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * These helpers only update the next ray. BSDF throughput is evaluated by
 * shadeBSDF so all path radiance updates happen in one place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine& rng);

__device__ void scatterMirror(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal);

__device__ void scatterRoughSpecular(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine& rng);

__device__ void scatterDielectric(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal,
    const Material& material,
    thrust::default_random_engine& rng);

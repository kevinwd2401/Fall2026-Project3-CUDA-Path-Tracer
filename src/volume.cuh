#pragma once

#include "sceneStructs.h"
#include <nanovdb/NanoVDB.h>
#include <thrust/random.h>
#include <cassert>
#include <cmath>

struct DeviceVolume
{
    const nanovdb::NanoGrid<float>* grid = nullptr;
    Volume medium;
};

__host__ __device__ inline bool volumeInterval(const Volume& volume, const Ray& ray,
    float maxDistance, float& enter, float& exit)
{
    enter = 0.0f;
    exit = maxDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (ray.direction[axis] == 0.0f)
        {
            if (ray.origin[axis] < volume.boundsMin[axis] ||
                ray.origin[axis] > volume.boundsMax[axis]) return false;
            continue;
        }
        const float a = (volume.boundsMin[axis] - ray.origin[axis]) / ray.direction[axis];
        const float b = (volume.boundsMax[axis] - ray.origin[axis]) / ray.direction[axis];
        enter = fmaxf(enter, fminf(a, b));
        exit = fminf(exit, fmaxf(a, b));
    }
    return enter < exit;
}

__device__ inline float volumeUniform(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    return fminf(1.0f - 1.0e-7f, fmaxf(1.0e-7f, uniform(rng)));
}

__device__ inline float sampleMajorantDistance(float u, float majorant)
{
    // Invert F(s) = 1 - exp(-majorant * s); log1p preserves precision near u=0.
    const float distance = -log1pf(-u) / majorant;
    assert(distance > 0.0f);
    return distance;
}

__device__ inline float realCollisionProbability(float sigmaT, float majorant)
{
    // Accept the fraction of majorant events belonging to the real medium.
    const float probability = sigmaT / majorant;

    assert(sigmaT >= 0.0f && sigmaT <= majorant && probability >= 0.0f && probability <= 1.0f);
    return probability;
}

template <typename Accessor>
__device__ inline float volumeExtinction(const DeviceVolume& volume,
    const Ray& ray, float t, const Accessor& accessor)
{
    const glm::vec3 p = ray.origin + t * ray.direction;
    // Undo the scene scale before applying NanoVDB's authored transform.
    // Keep ray distances and extinction in scene world units for both trackers.
    const glm::vec3 gridWorld = volume.medium.inverseScale == 1.0f ? p :
        volume.medium.scaleCenter + (p - volume.medium.scaleCenter) * volume.medium.inverseScale;
    const auto index = volume.grid->worldToIndex(nanovdb::Vec3f(gridWorld.x, gridWorld.y, gridWorld.z));
    // The active index domain is the physical medium boundary. This extra
    // test excludes the empty corners of its rotated world AABB.
    const auto& bounds = volume.grid->indexBBox();
    for (int axis = 0; axis < 3; ++axis)
        if (index[axis] < float(bounds.min()[axis]) - 1.0f ||
            index[axis] > float(bounds.max()[axis]) + 1.0f) return 0.0f;
    const nanovdb::Coord base{int(floorf(index[0])), int(floorf(index[1])), int(floorf(index[2]))};
    const glm::vec3 fraction(index[0] - base[0], index[1] - base[1], index[2] - base[2]);
    float density = 0.0f;
    for (int corner = 0; corner < 8; ++corner)
    {
        const int x = corner & 1, y = (corner >> 1) & 1, z = (corner >> 2) & 1;
        float value;
        // Treat inactive voxels/tiles as vacuum. This makes the active-value
        // maximum a valid majorant even if the file stores inactive values.
        if (accessor.probeValue(nanovdb::Coord(base[0] + x, base[1] + y, base[2] + z), value))
            density += value * (x ? fraction.x : 1.0f - fraction.x) *
                (y ? fraction.y : 1.0f - fraction.y) * (z ? fraction.z : 1.0f - fraction.z);
    }
    return fmaxf(0.0f, density) * volume.medium.extinction;
}

// Sample an extinction event. At a real event the integrator uses implicit
// absorption: beta *= sigma_s / sigma_t, then always samples a scattering
// direction. No-event survival already accounts for T; do not multiply beta by T.
__device__ inline bool sampleMedium(const DeviceVolume& volume, const Ray& ray,
    float enter, float exit, thrust::default_random_engine& rng, float& collisionT, float& logPdfRatio)
{
    if (volume.medium.majorant <= 0.0f) return false;
    auto accessor = volume.grid->getAccessor(); // per-thread cache
    float t = enter;
    while (true)
    {
        const float next = t + sampleMajorantDistance(volumeUniform(rng), volume.medium.majorant);
        t = next > t ? next : nextafterf(t, INFINITY);
        if (t >= exit) return false;
        const float sigmaT = volumeExtinction(volume, ray, t, accessor);
        const float realProbability = realCollisionProbability(sigmaT, volume.medium.majorant);
        if (volumeUniform(rng) < realProbability)
        {
            collisionT = t;
            return true;
        }
        logPdfRatio += log1pf(-realProbability);
        // Null collisions preserve throughput, direction, the last real vertex
        // and depth, while accumulating the distance-sampling MIS ratio below.
    }
}

// Unbiased transmittance estimate for NEE; same global majorant, independent RNG.
__device__ inline float mediumTransmittance(const DeviceVolume& volume, const Ray& ray,
    float enter, float exit, thrust::default_random_engine& rng, float& logPdfRatio)
{
    if (volume.medium.majorant <= 0.0f) return 1.0f;
    auto accessor = volume.grid->getAccessor();
    float transmittance = 1.0f;
    float t = enter;
    while (true)
    {
        const float next = t + sampleMajorantDistance(volumeUniform(rng), volume.medium.majorant);
        t = next > t ? next : nextafterf(t, INFINITY);
        if (t >= exit) return transmittance;
        const float sigmaT = volumeExtinction(volume, ray, t, accessor);
        const float realProbability = realCollisionProbability(sigmaT, volume.medium.majorant);
        transmittance *= 1.0f - realProbability;
        logPdfRatio += log1pf(-realProbability);
        if (transmittance == 0.0f) return 0.0f;
    }
}

// Convention: incoming follows the incident path ray, outgoing follows the
// new ray. Positive g concentrates samples around incoming (forward scattering).
__device__ inline float phaseHG(const glm::vec3& incoming, const glm::vec3& outgoing, float g)
{
    if (fabsf(g) < 1.0e-3f) g = 0.0f; // exactly match the uniform sampling branch
    const float cosine = glm::clamp(glm::dot(incoming, outgoing), -1.0f, 1.0f);
    const float denominator = 1.0f + g * g - 2.0f * g * cosine;
    return (1.0f - g * g) / (12.566370614359172f * denominator * sqrtf(denominator));
}

struct PhaseSample
{
    glm::vec3 direction;
    float value;
    float pdf;
};

__device__ inline PhaseSample samplePhaseHG(const glm::vec3& incoming, float g,
    thrust::default_random_engine& rng)
{
    const float u = volumeUniform(rng);
    float cosine = 1.0f - 2.0f * u;
    if (fabsf(g) >= 1.0e-3f)
    {
        // Invert the HG cosine distribution. Incoming points along the ray,
        // so positive g must yield E[cos(theta)] = g (forward scattering).
        const float gSquared = g * g;
        const float ratio = (1.0f - gSquared) / (1.0f + g - 2.0f * g * u);
        cosine = (1.0f + gSquared - ratio * ratio) / (2.0f * g);
    }
    cosine = glm::clamp(cosine, -1.0f, 1.0f);
    const float sine = sqrtf(fmaxf(0.0f, 1.0f - cosine * cosine));
    const float phi = 6.283185307179586f * volumeUniform(rng);
    const glm::vec3 helper = fabsf(incoming.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    const glm::vec3 tangent = glm::normalize(glm::cross(helper, incoming));
    const glm::vec3 bitangent = glm::cross(incoming, tangent);
    const glm::vec3 direction = glm::normalize(cosine * incoming +
        sine * (cosf(phi) * tangent + sinf(phi) * bitangent));
    const float pdf = phaseHG(incoming, direction, g);
    return { direction, pdf, pdf };
}

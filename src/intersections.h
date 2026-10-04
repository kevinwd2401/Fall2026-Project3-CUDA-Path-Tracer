#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cfloat>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(const glm::mat4& m, const glm::vec4& v)
{
    return glm::vec3(m * v);
}

// Cache reciprocals once per ray, rather than divide at every visited node.
// Zero directions are handled explicitly to avoid 0 * infinity on slab faces.
__host__ __device__ inline glm::vec3 rayInverseDirection(const Ray& ray)
{
    return glm::vec3(ray.direction.x == 0.0f ? 0.0f : 1.0f / ray.direction.x,
        ray.direction.y == 0.0f ? 0.0f : 1.0f / ray.direction.y,
        ray.direction.z == 0.0f ? 0.0f : 1.0f / ray.direction.z);
}

__host__ __device__ inline bool traversalBoundsTest(const Ray& ray,
    const glm::vec3& inverseDirection, const glm::vec3& lower,
    const glm::vec3& upper, float maxDistance)
{
    float nearT = 0.0f;
    float farT = maxDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (ray.direction[axis] == 0.0f)
        {
            if (ray.origin[axis] < lower[axis] || ray.origin[axis] > upper[axis]) return false;
            continue;
        }
        const float lo = lower[axis] - ray.origin[axis];
        const float hi = upper[axis] - ray.origin[axis];
        // Also avoid 0 * infinity if the reciprocal of a tiny direction overflows.
        const float a = lo == 0.0f ? 0.0f : lo * inverseDirection[axis];
        const float b = hi == 0.0f ? 0.0f : hi * inverseDirection[axis];
        nearT = fmaxf(nearT, fminf(a, b));
        farT = fminf(farT, fmaxf(a, b));
        if (nearT > farT) return false;
    }
    return true;
}

// Decide once per ray whether the branch-free slab test is safe. Parallel
// rays and overflowing reciprocals keep the robust face/zero handling above.
__host__ __device__ inline bool hasFiniteRayReciprocals(const glm::vec3& inverseDirection)
{
    const float x = fabsf(inverseDirection.x);
    const float y = fabsf(inverseDirection.y);
    const float z = fabsf(inverseDirection.z);
    return x > 0.0f && x <= FLT_MAX && y > 0.0f && y <= FLT_MAX && z > 0.0f && z <= FLT_MAX;
}

__host__ __device__ inline bool traversalBoundsTestFast(const Ray& ray,
    const glm::vec3& inverseDirection, const glm::vec3& lower,
    const glm::vec3& upper, float maxDistance)
{
    const glm::vec3 a = (lower - ray.origin) * inverseDirection;
    const glm::vec3 b = (upper - ray.origin) * inverseDirection;
    const float nearT = fmaxf(0.0f, fmaxf(fminf(a.x, b.x),
        fmaxf(fminf(a.y, b.y), fminf(a.z, b.z))));
    const float farT = fminf(maxDistance, fminf(fmaxf(a.x, b.x),
        fminf(fmaxf(a.y, b.y), fmaxf(a.z, b.z))));
    return nearT <= farT;
}

// Do not normalize the object-space direction: affine transforms preserve t.
// These distance-only tests are inlined into both closest-hit and any-hit code.
__host__ __device__ inline float boxDistanceTest(const Cube& box, const Ray& ray,
    float maxDistance)
{
    const glm::vec3 origin = multiplyMV(box.inverseTransform, glm::vec4(ray.origin, 1.0f));
    const glm::vec3 direction = multiplyMV(box.inverseTransform, glm::vec4(ray.direction, 0.0f));
    float nearT = -FLT_MAX;
    float farT = maxDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0.0f)
        {
            if (origin[axis] < -0.5f || origin[axis] > 0.5f) return -1.0f;
            continue;
        }
        const float a = (-0.5f - origin[axis]) / direction[axis];
        const float b = (0.5f - origin[axis]) / direction[axis];
        nearT = fmaxf(nearT, fminf(a, b));
        farT = fminf(farT, fmaxf(a, b));
        if (nearT > farT) return -1.0f;
    }
    const float t = nearT > 1e-4f ? nearT : farT;
    return t > 1e-4f && t < maxDistance ? t : -1.0f;
}

__host__ __device__ inline float sphereDistanceTest(const Sphere& sphere, const Ray& ray,
    float maxDistance)
{
    const glm::vec3 origin = multiplyMV(sphere.inverseTransform, glm::vec4(ray.origin, 1.0f));
    const glm::vec3 direction = multiplyMV(sphere.inverseTransform, glm::vec4(ray.direction, 0.0f));
    const float a = glm::dot(direction, direction);
    const float b = glm::dot(origin, direction);
    const float c = glm::dot(origin, origin) - 0.25f;
    const float discriminant = b * b - a * c;
    if (a <= 0.0f || discriminant < 0.0f) return -1.0f;
    // Stable quadratic roots, including tangent rays and origins on the surface.
    const float q = -b - copysignf(sqrtf(discriminant), b);
    const float t0 = q / a;
    const float t1 = q == 0.0f ? -b / a : c / q;
    const float nearT = fminf(t0, t1);
    const float farT = fmaxf(t0, t1);
    const float t = nearT > 1e-4f ? nearT : farT;
    return t > 1e-4f && t < maxDistance ? t : -1.0f;
}

__host__ __device__ inline float triangleDistanceTest(const glm::vec3& vertex,
    const glm::vec3& edge1, const glm::vec3& edge2, const Ray& ray,
    float maxDistance, glm::vec2& barycentrics)
{
    const glm::vec3 p = glm::cross(ray.direction, edge2);
    const float determinant = glm::dot(edge1, p);
    if (fabsf(determinant) < 1e-8f) return -1.0f;
    const float invDet = 1.0f / determinant;
    const glm::vec3 offset = ray.origin - vertex;
    const float u = glm::dot(offset, p) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const glm::vec3 q = glm::cross(offset, edge1);
    const float v = glm::dot(ray.direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    const float t = glm::dot(edge2, q) * invDet;
    if (!(t > 1e-4f && t < maxDistance)) return -1.0f;
    barycentrics = glm::vec2(u, v);
    return t;
}

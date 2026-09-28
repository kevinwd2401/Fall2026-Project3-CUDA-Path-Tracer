#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
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

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(const Ray& r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(const glm::mat4& m, const glm::vec4& v)
{
    return glm::vec3(m * v);
}

/**
 * Intersect a world-space axis-aligned bounding box.  maxDistance is the
 * nearest primitive hit already found, so nodes farther than it can be
 * skipped during BVH traversal.
 */
__host__ __device__ inline bool aabbIntersectionTest(
    const Ray& ray,
    const glm::vec3& boundsMin,
    const glm::vec3& boundsMax,
    float maxDistance,
    float& entryDistance)
{
    float tNear = -1.0e30f;
    float tFar = maxDistance;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.origin[axis];
        const float direction = ray.direction[axis];
        if (direction == 0.0f)
        {
            if (origin < boundsMin[axis] || origin > boundsMax[axis]) return false;
            continue;
        }
        float t0 = (boundsMin[axis] - origin) / direction;
        float t1 = (boundsMax[axis] - origin) / direction;
        if (t0 > t1)
        {
            const float temporary = t0;
            t0 = t1;
            t1 = temporary;
        }
        tNear = glm::max(tNear, t0);
        tFar = glm::min(tFar, t1);
        if (tNear > tFar) return false;
    }
    entryDistance = glm::max(tNear, 0.0f);
    return tFar >= 0.0f;
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

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    const Cube& box,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    const Sphere& sphere,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

/** Test a ray against a world-space triangle. */
__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside);

#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    const Cube& box, const Ray& r, glm::vec3& intersectionPoint,
    glm::vec3& normal, bool& outside)
{
    const float t = boxDistanceTest(box, r, FLT_MAX);
    if (t < 0.0f) return t;
    intersectionPoint = r.origin + t * r.direction;
    const glm::vec3 local = multiplyMV(box.inverseTransform, glm::vec4(intersectionPoint, 1.0f));
    glm::vec3 localNormal(0.0f);
    int axis = fabsf(local.x) >= fabsf(local.y) ? 0 : 1;
    if (fabsf(local.z) > fabsf(local[axis])) axis = 2;
    localNormal[axis] = local[axis] >= 0.0f ? 1.0f : -1.0f;
    normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(localNormal, 0.0f)));
    outside = glm::dot(r.direction, normal) < 0.0f;
    return t;
}

__host__ __device__ float sphereIntersectionTest(
    const Sphere& sphere, const Ray& r, glm::vec3& intersectionPoint,
    glm::vec3& normal, bool& outside)
{
    const float t = sphereDistanceTest(sphere, r, FLT_MAX);
    if (t < 0.0f) return t;
    intersectionPoint = r.origin + t * r.direction;
    const glm::vec3 local = multiplyMV(sphere.inverseTransform, glm::vec4(intersectionPoint, 1.0f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(local, 0.0f)));
    outside = glm::dot(r.direction, normal) < 0.0f;
    return t;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    const glm::vec3 edge1 = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    const glm::vec3 edge2 = triangle.triangleVertices[2] - triangle.triangleVertices[0];
    glm::vec2 barycentrics;
    const float t = triangleDistanceTest(triangle.triangleVertices[0], edge1, edge2,
        r, FLT_MAX, barycentrics);
    if (t < 0.0f) return t;
    const float u = barycentrics.x;
    const float v = barycentrics.y;
    const glm::vec3 faceNormal = glm::cross(edge1, edge2);
    intersectionPoint = r.origin + t * r.direction;
    uv = triangle.hasTextureCoordinates ?
        (1.0f - u - v) * triangle.triangleUVs[0] + u * triangle.triangleUVs[1] + v * triangle.triangleUVs[2] :
        glm::vec2(0.0f);
    if (triangle.hasVertexNormals)
    {
        const float w = 1.0f - u - v;
        normal = w * triangle.triangleNormals[0] +
            u * triangle.triangleNormals[1] + v * triangle.triangleNormals[2];
        if (glm::dot(normal, normal) < 1e-16f)
        {
            normal = glm::normalize(faceNormal);
        }
        else
        {
            normal = glm::normalize(normal);
        }
    }
    else
    {
        normal = glm::normalize(faceNormal);
    }
    outside = glm::dot(r.direction, normal) < 0.0f;
    return t;
}

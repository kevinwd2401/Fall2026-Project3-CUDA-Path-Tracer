#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Cube box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Sphere sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    // Moller-Trumbore in world space. glTF instances have already had their
    // node transforms applied by the host loader, so t is directly compatible
    // with the ray distances used by the rest of the kernel.
    const glm::vec3& p0 = triangle.triangleVertices[0];
    const glm::vec3& p1 = triangle.triangleVertices[1];
    const glm::vec3& p2 = triangle.triangleVertices[2];
    const glm::vec3 edge1 = p1 - p0;
    const glm::vec3 edge2 = p2 - p0;
    const glm::vec3 faceNormal = glm::cross(edge1, edge2);
    if (glm::dot(faceNormal, faceNormal) < 1e-16f)
    {
        return -1.0f;
    }

    const glm::vec3 pvec = glm::cross(r.direction, edge2);
    const float determinant = glm::dot(edge1, pvec);
    if (fabsf(determinant) < 1e-8f)
    {
        return -1.0f;
    }

    const float inverseDeterminant = 1.0f / determinant;
    const glm::vec3 tvec = r.origin - p0;
    const float u = glm::dot(tvec, pvec) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(r.direction, qvec) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f)
    {
        return -1.0f;
    }

    const float t = glm::dot(edge2, qvec) * inverseDeterminant;
    if (t <= 1e-4f)
    {
        return -1.0f;
    }

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

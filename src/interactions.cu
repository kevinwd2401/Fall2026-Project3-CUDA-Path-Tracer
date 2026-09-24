#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    normal = glm::normalize(normal);
    glm::vec3 wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.ray.direction = wi;
    pathSegment.ray.origin = intersect + 0.001f * normal;

}

__device__ void coordinateSystem(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    if (fabsf(normal.x) > fabsf(normal.z))
    {
        tangent = glm::normalize(glm::vec3(-normal.y, normal.x, 0.0f));
    }
    else
    {
        tangent = glm::normalize(glm::vec3(0.0f, -normal.z, normal.y));
    }
    bitangent = glm::normalize(glm::cross(normal, tangent));
}

__device__ glm::vec3 worldToLocal(const glm::vec3& normal, const glm::vec3& v)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    coordinateSystem(normal, tangent, bitangent);
    return glm::vec3(glm::dot(v, tangent), glm::dot(v, bitangent), glm::dot(v, normal));
}

__device__ glm::vec3 localToWorld(const glm::vec3& normal, const glm::vec3& v)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    coordinateSystem(normal, tangent, bitangent);
    return glm::normalize(v.x * tangent + v.y * bitangent + v.z * normal);
}

__device__ float absCosTheta(const glm::vec3& v)
{
    return fabsf(v.z);
}

__device__ bool sameHemisphere(const glm::vec3& a, const glm::vec3& b)
{
    return a.z * b.z > 0.0f;
}

__device__ void scatterMirror(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal)
{
    normal = glm::normalize(normal);
    glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
    if (glm::dot(incoming, normal) > 0.0f)
    {
        normal = -normal;
    }

    pathSegment.ray.direction = glm::normalize(glm::reflect(incoming, normal));
    pathSegment.ray.origin = intersect + 0.001f * normal;
}

__device__ glm::vec3 sampleTrowbridgeReitzWh(const glm::vec3& wo, float roughness, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u1 = glm::clamp(u01(rng), 0.0001f, 0.9999f);
    float u2 = u01(rng);
    roughness = glm::clamp(roughness, 0.001f, 1.0f);

    float tanTheta2 = roughness * roughness * u1 / (1.0f - u1);
    float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = TWO_PI * u2;

    glm::vec3 wh = glm::normalize(glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));
    return sameHemisphere(wo, wh) ? wh : -wh;
}

__device__ void scatterRoughSpecular(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine& rng)
{
    normal = glm::normalize(normal);
    glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
    if (glm::dot(incoming, normal) > 0.0f)
    {
        normal = -normal;
    }

    glm::vec3 wo = worldToLocal(normal, -incoming);
    if (wo.z == 0.0f)
    {
        scatterMirror(pathSegment, intersect, normal);
        return;
    }

    glm::vec3 wh = sampleTrowbridgeReitzWh(wo, roughness, rng);
    glm::vec3 wi = glm::reflect(-wo, wh);
    if (!sameHemisphere(wo, wi))
    {
        scatterMirror(pathSegment, intersect, normal);
        return;
    }

    float cosThetaO = absCosTheta(wo);
    float cosThetaI = absCosTheta(wi);
    float woDotWh = fabsf(glm::dot(wo, wh));
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f || woDotWh <= 0.0f)
    {
        scatterMirror(pathSegment, intersect, normal);
        return;
    }

    pathSegment.ray.direction = localToWorld(normal, wi);
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
}

// Returns the unpolarized Fresnel reflectance for an interface from etaI to
// etaT.  cosThetaI is always measured against the normal facing the incident
// medium, so this routine never has to infer which side of the surface a ray
// is on from a signed cosine.
__device__ float dielectricFresnel(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float etaRatio = etaI / etaT;
    float sinThetaT = etaRatio * sinThetaI;
    if (sinThetaT >= 1.0f)
    {
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

__device__ void scatterDielectric(
    PathSegment& pathSegment,
    const glm::vec3& intersect,
    glm::vec3 normal,
    const Material& material,
    thrust::default_random_engine& rng)
{
    normal = glm::normalize(normal);
    glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);

    float eta = material.indexOfRefraction > 0.0f ? material.indexOfRefraction : 1.55f;
    bool entering = glm::dot(incoming, normal) < 0.0f;
    glm::vec3 orientedNormal = entering ? normal : -normal;
    float etaI = entering ? 1.0f : eta;
    float etaT = entering ? eta : 1.0f;
    float etaRatio = etaI / etaT;

    // `orientedNormal` must face the incident medium.  This is especially
    // important at the silhouette: using the un-oriented geometric normal
    // makes the Fresnel and refraction calculations disagree about etaI/etaT.
    float cosTheta = glm::clamp(glm::dot(-incoming, orientedNormal), 0.0f, 1.0f);
    float sin2Theta = fmaxf(0.0f, 1.0f - cosTheta * cosTheta);
    bool totalInternalReflection = etaRatio * etaRatio * sin2Theta >= 1.0f;

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float reflectance = dielectricFresnel(cosTheta, etaI, etaT);
    float reflectionPdf = totalInternalReflection ? 1.0f : reflectance;
    glm::vec3 outgoing;

    if (totalInternalReflection || u01(rng) < reflectionPdf)
    {
        outgoing = glm::normalize(glm::reflect(incoming, orientedNormal));
    }
    else
    {
        float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - etaRatio * etaRatio * sin2Theta));
        outgoing = glm::normalize(etaRatio * incoming +
            (etaRatio * cosTheta - cosThetaT) * orientedNormal);
    }

    pathSegment.ray.direction = outgoing;

    // Offset on the outgoing side of the interface, along its normal.  An
    // offset along `outgoing` collapses to nearly zero in the normal direction
    // at grazing angles and causes self-intersections that appear as a bright
    // rim around glass.
    float outgoingSide = glm::dot(outgoing, orientedNormal) >= 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + 0.001f * outgoingSide * orientedNormal;
}

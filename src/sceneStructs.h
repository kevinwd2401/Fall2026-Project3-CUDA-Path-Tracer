#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    VOLUME
};

enum MaterialType
{
    MATERIAL_DIFFUSE = 0,
    MATERIAL_COOK_TORRANCE,
    MATERIAL_DIELECTRIC,
    MATERIAL_MIRROR,
    MATERIAL_MICROFACETS,
    MATERIAL_EMISSIVE,
    MATERIAL_TYPE_COUNT
};

enum AlphaMode
{
    ALPHA_OPAQUE = 0,
    ALPHA_MASK,
    ALPHA_BLEND
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// Single bounded, non-emissive medium, coefficients use inverse world units.
struct Volume
{
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    glm::vec3 scaleCenter{0.0f}; // world-space pivot; preserves the authored center
    float inverseScale = 1.0f; // scene world -> original grid world
    glm::vec3 albedo{0.9f}; // sigma_s / sigma_t, per color channel
    float extinction = 1.0f; // sigma_t = density * extinction
    float majorant = 0.0f;  // conservative global bound on sigma_t
    float g = 0.2f;
};

enum InteractionType { INTERACTION_MISS, INTERACTION_SURFACE, INTERACTION_MEDIUM };

struct PrimitiveRef
{
    GeomType type;
    int index;
};

// Each primitive list only stores the data needed by that geometry type.
// PrimitiveRef provides the stable scene-wide identity used by the BVH,
// intersections, and emissive-primitive list.
struct Cube
{
    int materialid;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Sphere
{
    int materialid;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

// glTF meshes are flattened into world-space triangles before upload.
struct Triangle
{
    int materialid;
    glm::vec3 triangleVertices[3];
    glm::vec3 triangleNormals[3];
    glm::vec2 triangleUVs[3];
    int hasVertexNormals;
    int hasTextureCoordinates;
};

// Device-only triangle layout. Traversal reads three independent position/
// edge arrays; interpolation and texture data are fetched only after a hit.
struct TriangleAttributes
{
    int materialid;
    glm::vec3 triangleNormals[3];
    glm::vec2 triangleUVs[3];
    int hasVertexNormals;
    int hasTextureCoordinates;
};

struct TriangleSoA
{
    const glm::vec3* vertices;
    const glm::vec3* edges1;
    const glm::vec3* edges2;
    const TriangleAttributes* attributes;
};

// A flat, depth-first CPU BVH node, split into bound and link arrays at upload.
// Leaves have primitiveCount > 0 and reference a range in
// Scene::bvhPrimitiveIndices.  Internal nodes have primitiveCount == 0.
//
// escapeIndex makes the tree stackless on the GPU: after a node (and all of
// its descendants) has been considered, traversal continues at escapeIndex.
// The root's escapeIndex is bvhNodes.size(), the traversal sentinel.
struct BVHNode
{
    glm::vec3 boundsMin;
    int firstPrimitive;
    glm::vec3 boundsMax;
    int primitiveCount;
    int escapeIndex;
};

struct Material
{
    glm::vec3 color;
    float alpha;
    glm::vec3 emission;
    MaterialType type;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasDielectric;
    float indexOfRefraction;
    float emittance;
    float metallic;
    float roughness;
    int baseColorTexture;
    int normalTexture;
    int metallicRoughnessTexture;
    float normalScale;
    AlphaMode alphaMode;
    float alphaCutoff;
};

// Texture texels are packed as RGBA8 in Scene::textureTexels. texelOffset
// points into that flat array so this POD descriptor can be copied directly
// to CUDA without the 4x expansion of a float RGBA image.
struct TextureInfo
{
    int width;
    int height;
    int texelOffset;
    int wrapS;
    int wrapT;
    int minFilter;
    int magFilter;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    // color is the path throughput.  Radiance is accumulated separately so a
    // direct-light sample does not terminate the path that generated it.
    glm::vec3 color;
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
    float previousScatteringPdf;
    bool previousEventWasDelta;
    // Alpha pass-throughs move ray.origin but must retain the last real vertex
    // for the complementary light PDF at an eventual emitter hit.
    glm::vec3 previousScatteringPoint;
    // log(product sigma_null / sigma_majorant) since the last real vertex.
    // Ratio of delta-tracking to ratio-tracking densities for the null history.
    float previousMediumLogPdfRatio;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  InteractionType type;
  float t;
  // Triangle barycentrics
  // Normals and UVs are reconstructed once in shading, outside traversal.
  glm::vec2 barycentrics;
  int primitiveIndex;
  // Candidate volume interval, independent of the closest surface hit.
  float volumeEnter;
  float volumeExit;
  bool hasVolumeInterval;
};

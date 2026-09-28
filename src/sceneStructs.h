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
    TRIANGLE
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

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // glTF meshes are flattened into world-space triangles before they are
    // copied to the device.  Keeping them in Geom lets the existing primitive
    // upload and nearest-hit loop serve analytic and mesh primitives alike.
    glm::vec3 triangleVertices[3];
    glm::vec3 triangleNormals[3];
    glm::vec2 triangleUVs[3];
    int hasVertexNormals;
    int hasTextureCoordinates;
};

// A flat, depth-first BVH node shared verbatim by the CPU builder and CUDA
// traversal code.  Leaves have primitiveCount > 0 and reference a range in
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
    float normalScale;
};

// Texture texels are packed into Scene::textureTexels.  texelOffset points
// into that flat array so this POD descriptor can be copied directly to CUDA.
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
    float previousBsdfPdf;
    bool previousBounceWasSpecular;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 surfaceUV;
  int materialId;
  int primitiveIndex;
};

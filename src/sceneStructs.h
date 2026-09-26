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
    int hasVertexNormals;
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
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

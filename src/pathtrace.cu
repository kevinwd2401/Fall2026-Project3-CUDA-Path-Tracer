#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "tonemapping.h"

#ifndef MATERIAL_SORT
#define MATERIAL_SORT 0
#endif
// Morton ordering schedules both closest-hit and shadow traversal by ray
// origin/direction. Material sorting remains independently available for shading.
#ifndef MORTON_SORT
#define MORTON_SORT 1
#endif
#define RUSSIAN_ROULETTE 1
#define RUSSIAN_ROULETTE_START_DEPTH 3
#define DEPTH_OF_FIELD 1
#define ERRORCHECK 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        const glm::vec3 pix = image[index] / static_cast<float>(iter);
        const glm::vec3 displayColor = acesFilmicTonemap(pix);

        glm::ivec3 color;
        color.x = glm::clamp((int)(displayColor.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(displayColor.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(displayColor.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static PrimitiveRef* dev_primitives = NULL;
static Cube* dev_cubes = NULL;
static Sphere* dev_spheres = NULL;
static glm::vec3* dev_triangleVertices = NULL;
static glm::vec3* dev_triangleEdges1 = NULL;
static glm::vec3* dev_triangleEdges2 = NULL;
static TriangleAttributes* dev_triangleAttributes = NULL;
static glm::vec3* dev_bvhBoundsMin = NULL;
static glm::vec3* dev_bvhBoundsMax = NULL;
static int3* dev_bvhLinks = NULL;
static int* dev_bvhPrimitiveIndices = NULL;
static Material* dev_materials = NULL;
static int* dev_lightPrimitives = NULL;
static int dev_lightPrimitiveCount = 0;
static glm::vec3* dev_environmentTexels = NULL;
static float* dev_environmentAliasProbability = NULL;
static int* dev_environmentAliasIndex = NULL;
static float* dev_environmentPdfSolidAngle = NULL;
static TextureInfo* dev_textures = NULL;
static glm::vec4* dev_textureTexels = NULL;
static int dev_textureCount = 0;

struct DeviceEnvironmentMap
{
    const glm::vec3* texels = NULL;
    const float* aliasProbability = NULL;
    const int* aliasIndex = NULL;
    const float* pdfSolidAngle = NULL;
    int width = 0;
    int height = 0;

    __host__ __device__ bool valid() const
    {
        return texels != NULL && aliasProbability != NULL && aliasIndex != NULL &&
            pdfSolidAngle != NULL && width > 0 && height > 0;
    }
};

static DeviceEnvironmentMap dev_environment;

struct DeviceTextureStore
{
    const TextureInfo* textures = NULL;
    const glm::vec4* texels = NULL;
    int textureCount = 0;

    __host__ __device__ bool validTextureIndex(int textureIndex) const
    {
        return textures != NULL && texels != NULL && textureIndex >= 0 && textureIndex < textureCount;
    }
};

static DeviceTextureStore dev_textureStore;

struct DevicePrimitiveStore
{
    const PrimitiveRef* primitives = NULL;
    const Cube* cubes = NULL;
    const Sphere* spheres = NULL;
    TriangleSoA triangles{};
};

static DevicePrimitiveStore dev_primitiveStore;

// Uniform, read-only descriptors live in constant memory so each traversal
// thread does not carry a large store argument. Node bounds are separate from
// leaf/topology metadata; the host builder retains its convenient AoS layout.
struct TraversalData
{
    const PrimitiveRef* primitives;
    const Cube* cubes;
    const Sphere* spheres;
    const glm::vec3* vertices;
    const glm::vec3* edges1;
    const glm::vec3* edges2;
    const glm::vec3* boundsMin;
    const glm::vec3* boundsMax;
    const int3* links; // firstPrimitive, primitiveCount, escapeIndex
    const int* primitiveIndices;
    int nodeCount;
};

__constant__ TraversalData traversalData;

static glm::vec3 mortonBoundsMin(0.0f);
static glm::vec3 mortonInverseExtent(0.0f);

// A shading thread owns the slot at its current path index. Inactive entries
// are compacted away before the dedicated shadow traversal kernel.
struct ShadowRay
{
    Ray ray;
    glm::vec3 contribution;
    float maxDistance;
    int pathIndex;
    int active;
    int occluded;
};

static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadowRay* dev_shadowRays = NULL;
static int* dev_materialSortKeys = NULL;
static unsigned int* dev_mortonSortKeys = NULL;
static int* dev_rayOrder = NULL;

template <typename T>
T* uploadArray(const std::vector<T>& values)
{
    T* device = NULL;
    if (!values.empty())
    {
        cudaMalloc(&device, values.size() * sizeof(T));
        cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    return device;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(PrimitiveRef));
    cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(PrimitiveRef), cudaMemcpyHostToDevice);
    if (!scene->cubes.empty())
    {
        cudaMalloc(&dev_cubes, scene->cubes.size() * sizeof(Cube));
        cudaMemcpy(dev_cubes, scene->cubes.data(), scene->cubes.size() * sizeof(Cube), cudaMemcpyHostToDevice);
    }
    if (!scene->spheres.empty())
    {
        cudaMalloc(&dev_spheres, scene->spheres.size() * sizeof(Sphere));
        cudaMemcpy(dev_spheres, scene->spheres.data(), scene->spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
    }
    if (!scene->triangles.empty())
    {
        std::vector<glm::vec3> vertices, edges1, edges2;
        std::vector<TriangleAttributes> attributes;
        vertices.reserve(scene->triangles.size());
        edges1.reserve(scene->triangles.size());
        edges2.reserve(scene->triangles.size());
        attributes.reserve(scene->triangles.size());
        for (const Triangle& triangle : scene->triangles)
        {
            vertices.push_back(triangle.triangleVertices[0]);
            edges1.push_back(triangle.triangleVertices[1] - triangle.triangleVertices[0]);
            edges2.push_back(triangle.triangleVertices[2] - triangle.triangleVertices[0]);
            TriangleAttributes attribute{};
            attribute.materialid = triangle.materialid;
            attribute.hasVertexNormals = triangle.hasVertexNormals;
            attribute.hasTextureCoordinates = triangle.hasTextureCoordinates;
            for (int corner = 0; corner < 3; ++corner)
            {
                attribute.triangleNormals[corner] = triangle.triangleNormals[corner];
                attribute.triangleUVs[corner] = triangle.triangleUVs[corner];
            }
            attributes.push_back(attribute);
        }
        dev_triangleVertices = uploadArray(vertices);
        dev_triangleEdges1 = uploadArray(edges1);
        dev_triangleEdges2 = uploadArray(edges2);
        dev_triangleAttributes = uploadArray(attributes);
    }
    dev_primitiveStore = { dev_primitives, dev_cubes, dev_spheres,
        { dev_triangleVertices, dev_triangleEdges1, dev_triangleEdges2, dev_triangleAttributes } };

    mortonBoundsMin = glm::vec3(0.0f);
    mortonInverseExtent = glm::vec3(0.0f);
    if (!scene->bvhNodes.empty())
    {
        std::vector<glm::vec3> lower, upper;
        std::vector<int3> links;
        lower.reserve(scene->bvhNodes.size());
        upper.reserve(scene->bvhNodes.size());
        links.reserve(scene->bvhNodes.size());
        for (const BVHNode& node : scene->bvhNodes)
        {
            lower.push_back(node.boundsMin);
            upper.push_back(node.boundsMax);
            links.push_back(make_int3(node.firstPrimitive, node.primitiveCount, node.escapeIndex));
        }
        dev_bvhBoundsMin = uploadArray(lower);
        dev_bvhBoundsMax = uploadArray(upper);
        dev_bvhLinks = uploadArray(links);
        dev_bvhPrimitiveIndices = uploadArray(scene->bvhPrimitiveIndices);
        mortonBoundsMin = scene->bvhNodes[0].boundsMin;
        const glm::vec3 extent = scene->bvhNodes[0].boundsMax - mortonBoundsMin;
        for (int axis = 0; axis < 3; ++axis)
            mortonInverseExtent[axis] = extent[axis] > 1e-8f ? 1.0f / extent[axis] : 0.0f;
    }
    const TraversalData traversal = { dev_primitives, dev_cubes, dev_spheres,
        dev_triangleVertices, dev_triangleEdges1, dev_triangleEdges2,
        dev_bvhBoundsMin, dev_bvhBoundsMax, dev_bvhLinks, dev_bvhPrimitiveIndices,
        static_cast<int>(scene->bvhNodes.size()) };
    cudaMemcpyToSymbol(traversalData, &traversal, sizeof(traversal));

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    dev_lightPrimitiveCount = static_cast<int>(scene->emissivePrimitives.size());
    if (dev_lightPrimitiveCount > 0)
    {
        cudaMalloc(&dev_lightPrimitives, dev_lightPrimitiveCount * sizeof(int));
        cudaMemcpy(dev_lightPrimitives, scene->emissivePrimitives.data(),
            dev_lightPrimitiveCount * sizeof(int), cudaMemcpyHostToDevice);
    }

    if (scene->environment.valid())
    {
        const size_t environmentPixelCount = scene->environment.texels.size();
        cudaMalloc(&dev_environmentTexels, environmentPixelCount * sizeof(glm::vec3));
        cudaMalloc(&dev_environmentAliasProbability, environmentPixelCount * sizeof(float));
        cudaMalloc(&dev_environmentAliasIndex, environmentPixelCount * sizeof(int));
        cudaMalloc(&dev_environmentPdfSolidAngle, environmentPixelCount * sizeof(float));
        cudaMemcpy(dev_environmentTexels, scene->environment.texels.data(),
            environmentPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_environmentAliasProbability, scene->environment.aliasProbability.data(),
            environmentPixelCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_environmentAliasIndex, scene->environment.aliasIndex.data(),
            environmentPixelCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_environmentPdfSolidAngle, scene->environment.pdfSolidAngle.data(),
            environmentPixelCount * sizeof(float), cudaMemcpyHostToDevice);
        dev_environment = { dev_environmentTexels, dev_environmentAliasProbability,
            dev_environmentAliasIndex, dev_environmentPdfSolidAngle,
            scene->environment.width, scene->environment.height };
    }

    dev_textureCount = static_cast<int>(scene->textures.size());
    if (dev_textureCount > 0 && !scene->textureTexels.empty())
    {
        cudaMalloc(&dev_textures, dev_textureCount * sizeof(TextureInfo));
        cudaMalloc(&dev_textureTexels, scene->textureTexels.size() * sizeof(glm::vec4));
        cudaMemcpy(dev_textures, scene->textures.data(), dev_textureCount * sizeof(TextureInfo), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_textureTexels, scene->textureTexels.data(),
            scene->textureTexels.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);
        dev_textureStore = { dev_textures, dev_textureTexels, dev_textureCount };
    }

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_shadowRays, pixelcount * sizeof(ShadowRay));

#if MATERIAL_SORT
    cudaMalloc(&dev_materialSortKeys, pixelcount * sizeof(int));
#endif
#if MORTON_SORT
    cudaMalloc(&dev_mortonSortKeys, pixelcount * sizeof(unsigned int));
    cudaMalloc(&dev_rayOrder, pixelcount * sizeof(int));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_primitives);
    cudaFree(dev_cubes);
    cudaFree(dev_spheres);
    cudaFree(dev_triangleVertices);
    cudaFree(dev_triangleEdges1);
    cudaFree(dev_triangleEdges2);
    cudaFree(dev_triangleAttributes);
    cudaFree(dev_bvhBoundsMin);
    cudaFree(dev_bvhBoundsMax);
    cudaFree(dev_bvhLinks);
    cudaFree(dev_bvhPrimitiveIndices);
    cudaFree(dev_materials);
    cudaFree(dev_lightPrimitives);
    cudaFree(dev_environmentTexels);
    cudaFree(dev_environmentAliasProbability);
    cudaFree(dev_environmentAliasIndex);
    cudaFree(dev_environmentPdfSolidAngle);
    cudaFree(dev_textures);
    cudaFree(dev_textureTexels);
    cudaFree(dev_intersections);
    cudaFree(dev_shadowRays);
    cudaFree(dev_materialSortKeys);
    cudaFree(dev_mortonSortKeys);
    cudaFree(dev_rayOrder);
    dev_environmentTexels = NULL;
    dev_environmentAliasProbability = NULL;
    dev_environmentAliasIndex = NULL;
    dev_environmentPdfSolidAngle = NULL;
    dev_environment = DeviceEnvironmentMap{};
    dev_textures = NULL;
    dev_textureTexels = NULL;
    dev_textureCount = 0;
    dev_textureStore = DeviceTextureStore{};
    dev_primitives = NULL;
    dev_cubes = NULL;
    dev_spheres = NULL;
    dev_triangleVertices = NULL;
    dev_triangleEdges1 = NULL;
    dev_triangleEdges2 = NULL;
    dev_triangleAttributes = NULL;
    dev_primitiveStore = DevicePrimitiveStore{};
    dev_shadowRays = NULL;
    dev_image = NULL;
    dev_paths = NULL;
    dev_intersections = NULL;
    dev_bvhBoundsMin = NULL;
    dev_bvhBoundsMax = NULL;
    dev_bvhLinks = NULL;
    dev_bvhPrimitiveIndices = NULL;
    dev_materials = NULL;
    dev_lightPrimitives = NULL;
    dev_lightPrimitiveCount = 0;
    dev_materialSortKeys = NULL;
    dev_mortonSortKeys = NULL;
    dev_rayOrder = NULL;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, const float focusDistance, const float lensRadius)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x + cam.resolution.x * y, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

#if DEPTH_OF_FIELD

		glm::vec3 focalPoint = cam.position + cam.view * focusDistance;

        float r = lensRadius * sqrtf(u01(rng));
        float theta = TWO_PI * u01(rng);

        glm::vec3 aperturePt = cam.position
            + r * cosf(theta) * cam.right
            + r * sinf(theta) * cam.up;

        segment.ray.origin = aperturePt;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // implement antialiasing by jittering the ray

		segment.ray.direction = glm::normalize((focalPoint
            - cam.right * focusDistance * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
            - cam.up * focusDistance * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng)) - aperturePt)
        );
#else
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        segment.ray.direction = glm::normalize((cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng)))
        );
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.radiance = glm::vec3(0.0f);
        // Camera rays are not generated by a BSDF, so an emitter visible
        // directly through the lens must not receive a light-sampling weight.
        segment.previousBsdfPdf = 0.0f;
        segment.previousBounceWasSpecular = true;
    }
}

// Distance-only primitive dispatch.
__device__ __forceinline__ float intersectPrimitiveDistance(const PrimitiveRef& primitive,
    const Ray& ray, float maxDistance, glm::vec2& barycentrics)
{
    switch (primitive.type)
    {
    case CUBE:
        return boxDistanceTest(traversalData.cubes[primitive.index], ray, maxDistance);
    case SPHERE:
        return sphereDistanceTest(traversalData.spheres[primitive.index], ray, maxDistance);
    case TRIANGLE:
        return triangleDistanceTest(traversalData.vertices[primitive.index],
            traversalData.edges1[primitive.index], traversalData.edges2[primitive.index],
            ray, maxDistance, barycentrics);
    }
    return -1.0f;
}

__device__ int primitiveMaterialId(const PrimitiveRef& primitive, const DevicePrimitiveStore& primitiveStore)
{
    switch (primitive.type)
    {
    case CUBE: return primitiveStore.cubes[primitive.index].materialid;
    case SPHERE: return primitiveStore.spheres[primitive.index].materialid;
    case TRIANGLE: return primitiveStore.triangles.attributes[primitive.index].materialid;
    }
    return -1;
}

__global__ void computeIntersections(int numPaths, const PathSegment* pathSegments,
    const int* rayOrder, ShadeableIntersection* intersections)
{
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= numPaths) return;
    const int pathIndex = rayOrder ? rayOrder[lane] : lane;
    // Load only the six ray floats, never the full path/throughput payload.
    const Ray ray = pathSegments[pathIndex].ray;
    const glm::vec3 inverseDirection = rayInverseDirection(ray);
    float closest = FLT_MAX;
    int hitPrimitive = -1;
    glm::vec2 hitBarycentrics(0.0f);
    int nodeIndex = 0;
    while (nodeIndex < traversalData.nodeCount)
    {
        if (!traversalBoundsTest(ray, inverseDirection, traversalData.boundsMin[nodeIndex],
            traversalData.boundsMax[nodeIndex], closest))
        {
            nodeIndex = traversalData.links[nodeIndex].z;
            continue;
        }
        const int3 links = traversalData.links[nodeIndex];
        if (links.y == 0)
        {
            ++nodeIndex;
            continue;
        }
        for (int offset = 0; offset < links.y; ++offset)
        {
            const int primitiveIndex = traversalData.primitiveIndices[links.x + offset];
            glm::vec2 barycentrics(0.0f);
            const float t = intersectPrimitiveDistance(traversalData.primitives[primitiveIndex],
                ray, closest, barycentrics);
            if (t > 0.0f)
            {
                closest = t;
                hitPrimitive = primitiveIndex;
                hitBarycentrics = barycentrics;
            }
        }
        nodeIndex = links.z;
    }
    ShadeableIntersection& hit = intersections[pathIndex];
    hit.t = hitPrimitive < 0 ? -1.0f : closest;
    hit.primitiveIndex = hitPrimitive;
    hit.barycentrics = hitBarycentrics;
}

__device__ void bsdfCoordinateSystem(
    const glm::vec3& normal,
    glm::vec3& tangent,
    glm::vec3& bitangent)
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

__device__ glm::vec3 bsdfWorldToLocal(const glm::vec3& normal, const glm::vec3& v)
{
    glm::vec3 tangent;
    glm::vec3 bitangent;
    bsdfCoordinateSystem(normal, tangent, bitangent);
    return glm::vec3(glm::dot(v, tangent), glm::dot(v, bitangent), glm::dot(v, normal));
}

__device__ float bsdfTrowbridgeReitzLambda(const glm::vec3& w, float roughness)
{
    if (fabsf(w.z) <= 0.0f)
    {
        return 0.0f;
    }

    float absTanTheta = sqrtf(fmaxf(0.0f, 1.0f - w.z * w.z)) / fabsf(w.z);
    if (isinf(absTanTheta))
    {
        return 0.0f;
    }

    float alphaTanTheta = roughness * absTanTheta;
    return 0.5f * (-1.0f + sqrtf(1.0f + alphaTanTheta * alphaTanTheta));
}

__device__ glm::vec3 roughSpecularThroughput(
    const glm::vec3& incoming,
    const glm::vec3& outgoing,
    glm::vec3 normal,
    const Material& material,
    float roughness)
{
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f)
    {
        normal = -normal;
    }

    glm::vec3 wo = bsdfWorldToLocal(normal, -incoming);
    const float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
    // MICROFACET_REFL lobe uses albedo directly with F = 1.
    const glm::vec3 f0 = material.type == MATERIAL_MICROFACETS ? material.color :
        glm::mix(glm::vec3(0.04f), material.color, metallic);
    if (glm::length2(outgoing - glm::reflect(incoming, normal)) < 1e-10f)
    {
        return f0;
    }

    glm::vec3 wi = bsdfWorldToLocal(normal, outgoing);
    glm::vec3 wh = glm::normalize(wo + wi);
    float cosThetaO = fabsf(wo.z);
    float absCosThetaH = fabsf(wh.z);
    float woDotWh = fabsf(glm::dot(wo, wh));

    if (cosThetaO <= 0.0f || absCosThetaH <= 0.0f || woDotWh <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    float lambdaO = bsdfTrowbridgeReitzLambda(wo, roughness);
    float lambdaI = bsdfTrowbridgeReitzLambda(wi, roughness);
    float G = 1.0f / (1.0f + lambdaO + lambdaI);
    return f0 * (G * woDotWh / (cosThetaO * absCosThetaH));
}

struct LightSurfaceSample
{
    glm::vec3 position;
    glm::vec3 normal;
    float pdfArea;
    bool valid;
};

__device__ glm::vec3 emittedRadiance(const Material& material)
{
    if (fmaxf(material.emission.x, fmaxf(material.emission.y, material.emission.z)) > 0.0f)
    {
        return material.emission;
    }
    return material.color * material.emittance;
}

__device__ glm::vec3 lightSurfaceNormal(const PrimitiveRef& primitive,
    const DevicePrimitiveStore& primitiveStore, const glm::vec3& point)
{
    if (primitive.type == TRIANGLE)
    {
        return glm::normalize(glm::cross(
            primitiveStore.triangles.edges1[primitive.index],
            primitiveStore.triangles.edges2[primitive.index]));
    }

    if (primitive.type == SPHERE)
    {
        const Sphere& sphere = primitiveStore.spheres[primitive.index];
        const glm::vec3 localPoint = multiplyMV(sphere.inverseTransform, glm::vec4(point, 1.0f));
        return glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(glm::normalize(localPoint), 0.0f)));
    }

    const Cube& cube = primitiveStore.cubes[primitive.index];
    const glm::vec3 localPoint = multiplyMV(cube.inverseTransform, glm::vec4(point, 1.0f));
    glm::vec3 localNormal(0.0f);
    const glm::vec3 absolutePoint(fabsf(localPoint.x), fabsf(localPoint.y), fabsf(localPoint.z));
    if (absolutePoint.x >= absolutePoint.y && absolutePoint.x >= absolutePoint.z)
        localNormal.x = localPoint.x >= 0.0f ? 1.0f : -1.0f;
    else if (absolutePoint.y >= absolutePoint.z)
        localNormal.y = localPoint.y >= 0.0f ? 1.0f : -1.0f;
    else
        localNormal.z = localPoint.z >= 0.0f ? 1.0f : -1.0f;
    return glm::normalize(multiplyMV(cube.invTranspose, glm::vec4(localNormal, 0.0f)));
}

__device__ float primitiveSurfaceAreaPdf(const PrimitiveRef& primitive,
    const DevicePrimitiveStore& primitiveStore, const glm::vec3& point)
{
    if (primitive.type == TRIANGLE)
    {
        const float area = 0.5f * glm::length(glm::cross(
            primitiveStore.triangles.edges1[primitive.index],
            primitiveStore.triangles.edges2[primitive.index]));
        return area > EPSILON ? 1.0f / area : 0.0f;
    }

    if (primitive.type == SPHERE)
    {
        const Sphere& sphere = primitiveStore.spheres[primitive.index];
        const glm::vec3 localPoint = multiplyMV(sphere.inverseTransform, glm::vec4(point, 1.0f));
        if (glm::length2(localPoint) <= EPSILON) return 0.0f;
        const glm::vec3 localNormal = glm::normalize(localPoint);
        const float jacobian = fabsf(glm::determinant(glm::mat3(sphere.transform))) *
            glm::length(multiplyMV(sphere.invTranspose, glm::vec4(localNormal, 0.0f)));
        // The unit-local sphere has radius 0.5, and therefore area PI.
        return jacobian > EPSILON ? 1.0f / (PI * jacobian) : 0.0f;
    }

    const Cube& cube = primitiveStore.cubes[primitive.index];
    const glm::vec3 xAxis = multiplyMV(cube.transform, glm::vec4(1, 0, 0, 0));
    const glm::vec3 yAxis = multiplyMV(cube.transform, glm::vec4(0, 1, 0, 0));
    const glm::vec3 zAxis = multiplyMV(cube.transform, glm::vec4(0, 0, 1, 0));
    const float area = 2.0f * (glm::length(glm::cross(yAxis, zAxis)) +
        glm::length(glm::cross(xAxis, zAxis)) + glm::length(glm::cross(xAxis, yAxis)));
    return area > EPSILON ? 1.0f / area : 0.0f;
}

__device__ LightSurfaceSample sampleLightSurface(const PrimitiveRef& primitive,
    const DevicePrimitiveStore& primitiveStore, thrust::default_random_engine& rng)
{
    LightSurfaceSample sample{};
    sample.valid = false;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    if (primitive.type == TRIANGLE)
    {
        const TriangleSoA& triangles = primitiveStore.triangles;
        const float rootU = sqrtf(u01(rng));
        const float v = u01(rng);
        sample.position = triangles.vertices[primitive.index] +
            rootU * (1.0f - v) * triangles.edges1[primitive.index] +
            rootU * v * triangles.edges2[primitive.index];
        sample.normal = lightSurfaceNormal(primitive, primitiveStore, sample.position);
    }
    else if (primitive.type == SPHERE)
    {
        const Sphere& sphere = primitiveStore.spheres[primitive.index];
        const float z = 1.0f - 2.0f * u01(rng);
        const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = TWO_PI * u01(rng);
        const glm::vec3 localNormal(radial * cosf(phi), radial * sinf(phi), z);
        sample.position = multiplyMV(sphere.transform, glm::vec4(0.5f * localNormal, 1.0f));
        sample.normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(localNormal, 0.0f)));
    }
    else
    {
        const Cube& cube = primitiveStore.cubes[primitive.index];
        const glm::vec3 xAxis = multiplyMV(cube.transform, glm::vec4(1, 0, 0, 0));
        const glm::vec3 yAxis = multiplyMV(cube.transform, glm::vec4(0, 1, 0, 0));
        const glm::vec3 zAxis = multiplyMV(cube.transform, glm::vec4(0, 0, 1, 0));
        const float faceAreas[3] = { glm::length(glm::cross(yAxis, zAxis)),
            glm::length(glm::cross(xAxis, zAxis)), glm::length(glm::cross(xAxis, yAxis)) };
        const float totalArea = 2.0f * (faceAreas[0] + faceAreas[1] + faceAreas[2]);
        if (totalArea <= EPSILON) return sample;
        float chosenArea = u01(rng) * totalArea;
        int axis = 0;
        int sign = 1;
        for (int candidate = 0; candidate < 3; ++candidate)
        {
            const float pairArea = 2.0f * faceAreas[candidate];
            if (chosenArea < pairArea)
            {
                axis = candidate;
                sign = chosenArea < faceAreas[candidate] ? 1 : -1;
                break;
            }
            chosenArea -= pairArea;
        }
        glm::vec3 localPoint(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f);
        localPoint[axis] = 0.5f * static_cast<float>(sign);
        sample.position = multiplyMV(cube.transform, glm::vec4(localPoint, 1.0f));
        glm::vec3 localNormal(0.0f);
        localNormal[axis] = static_cast<float>(sign);
        sample.normal = glm::normalize(multiplyMV(cube.invTranspose, glm::vec4(localNormal, 0.0f)));
    }

    sample.pdfArea = primitiveSurfaceAreaPdf(primitive, primitiveStore, sample.position);
    sample.valid = sample.pdfArea > 0.0f;
    return sample;
}

__device__ float lightPdfSolidAngle(const PrimitiveRef& light, const DevicePrimitiveStore& primitiveStore,
    const glm::vec3& referencePoint, const glm::vec3& lightPoint, const glm::vec3& lightNormal, int lightCount)
{
    if (lightCount <= 0) return 0.0f;
    const glm::vec3 toLight = lightPoint - referencePoint;
    const float distance2 = glm::dot(toLight, toLight);
    if (distance2 <= EPSILON) return 0.0f;
    const float lightCosine = glm::dot(lightNormal, -glm::normalize(toLight));
    const float areaPdf = primitiveSurfaceAreaPdf(light, primitiveStore, lightPoint);
    return lightCosine > EPSILON ? distance2 * areaPdf /
        (lightCosine * static_cast<float>(lightCount)) : 0.0f;
}

// Lookup and sampling: Equirectangular maps use +Y at the top edge, and wrap around the X/Z axis.
__device__ int environmentTexelIndex(const DeviceEnvironmentMap& environment,
    const glm::vec3& direction)
{
    const glm::vec3 normalizedDirection = glm::normalize(direction);
    float u = atan2f(normalizedDirection.z, normalizedDirection.x) / TWO_PI + 0.5f;
    u -= floorf(u);
    const float v = acosf(glm::clamp(normalizedDirection.y, -1.0f, 1.0f)) / PI;
    const int x = min(environment.width - 1, static_cast<int>(u * environment.width));
    const int y = min(environment.height - 1, static_cast<int>(v * environment.height));
    return y * environment.width + x;
}

__device__ glm::vec3 environmentRadiance(const DeviceEnvironmentMap& environment,
    const glm::vec3& direction)
{
    return environment.valid() ? environment.texels[environmentTexelIndex(environment, direction)] :
        glm::vec3(0.0f);
}

__device__ float environmentPdf(const DeviceEnvironmentMap& environment,
    const glm::vec3& direction)
{
    return environment.valid() ? environment.pdfSolidAngle[environmentTexelIndex(environment, direction)] : 0.0f;
}

struct EnvironmentSample
{
    glm::vec3 direction;
    glm::vec3 radiance;
    float pdfSolidAngle;
    bool valid;
};

__device__ EnvironmentSample sampleEnvironment(const DeviceEnvironmentMap& environment,
    thrust::default_random_engine& rng)
{
    EnvironmentSample sample{};
    sample.valid = false;
    if (!environment.valid()) return sample;

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int texelCount = environment.width * environment.height;
    const float aliasSample = fminf(u01(rng), 0.99999994f) * texelCount;
    const int column = min(static_cast<int>(aliasSample), texelCount - 1);
    const float coinFlip = aliasSample - column;
    const int texelIndex = coinFlip < environment.aliasProbability[column] ?
        column : environment.aliasIndex[column];
    const int x = texelIndex % environment.width;
    const int y = texelIndex / environment.width;
    const float u = (static_cast<float>(x) + u01(rng)) / static_cast<float>(environment.width);
    const float v = (static_cast<float>(y) + u01(rng)) / static_cast<float>(environment.height);
    const float phi = TWO_PI * (u - 0.5f);
    const float theta = PI * v;
    const float sinTheta = sinf(theta);
    sample.direction = glm::vec3(cosf(phi) * sinTheta, cosf(theta), sinf(phi) * sinTheta);
    sample.radiance = environment.texels[texelIndex];
    sample.pdfSolidAngle = environment.pdfSolidAngle[texelIndex];
    sample.valid = sample.pdfSolidAngle > 0.0f;
    return sample;
}

__device__ __forceinline__ bool isOccluded(const Ray& ray, float maxDistance)
{
    const glm::vec3 inverseDirection = rayInverseDirection(ray);
    int nodeIndex = 0;
    while (nodeIndex < traversalData.nodeCount)
    {
        if (!traversalBoundsTest(ray, inverseDirection, traversalData.boundsMin[nodeIndex],
            traversalData.boundsMax[nodeIndex], maxDistance))
        {
            nodeIndex = traversalData.links[nodeIndex].z;
            continue;
        }
        const int3 links = traversalData.links[nodeIndex];
        if (links.y == 0)
        {
            ++nodeIndex;
            continue;
        }
        for (int offset = 0; offset < links.y; ++offset)
        {
            const PrimitiveRef& primitive = traversalData.primitives[
                traversalData.primitiveIndices[links.x + offset]];
            glm::vec2 unusedBarycentrics;
            if (intersectPrimitiveDistance(primitive, ray, maxDistance, unusedBarycentrics) > 0.0f)
                return true;
        }
        nodeIndex = links.z;
    }
    return false;
}

__device__ float wrapTextureCoordinate(float coordinate, int wrapMode)
{
    // glTF sampler constants: REPEAT = 10497, CLAMP_TO_EDGE = 33071,
    // MIRRORED_REPEAT = 33648.  Unknown modes fall back to REPEAT.
    if (wrapMode == 33071) return glm::clamp(coordinate, 0.0f, 1.0f);
    const float cell = floorf(coordinate);
    const float fraction = coordinate - cell;
    if (wrapMode == 33648)
    {
        const int cellIndex = static_cast<int>(cell);
        return (cellIndex % 2 == 0) ? fraction : 1.0f - fraction;
    }
    return fraction;
}

__device__ int wrapTextureTexel(int texel, int size, int wrapMode)
{
    if (wrapMode == 33071) return max(0, min(texel, size - 1));
    if (wrapMode == 33648)
    {
        const int period = 2 * size;
        int mirrored = texel % period;
        if (mirrored < 0) mirrored += period;
        return mirrored < size ? mirrored : period - 1 - mirrored;
    }
    int repeated = texel % size;
    return repeated < 0 ? repeated + size : repeated;
}

__device__ bool textureUsesLinearFilter(const TextureInfo& texture)
{
    const int filter = texture.magFilter >= 0 ? texture.magFilter : texture.minFilter;
    return filter == 9729 || filter == 9985 || filter == 9987; // LINEAR variants
}

__device__ glm::vec4 sampleTexture(const TextureInfo& texture, const glm::vec2& uv,
    const DeviceTextureStore& textureStore)
{
    const float u = wrapTextureCoordinate(uv.x, texture.wrapS);
    const float v = wrapTextureCoordinate(uv.y, texture.wrapT);
    const auto texelAt = [&](int x, int y) {
        x = wrapTextureTexel(x, texture.width, texture.wrapS);
        y = wrapTextureTexel(y, texture.height, texture.wrapT);
        return textureStore.texels[texture.texelOffset + y * texture.width + x];
    };

    // glTF UV (0, 0) corresponds to the first decoded image row, so V is not
    // flipped here.  This also makes the same sampler work for color and normal maps.
    if (!textureUsesLinearFilter(texture))
    {
        return texelAt(static_cast<int>(floorf(u * texture.width)),
            static_cast<int>(floorf(v * texture.height)));
    }

    const float x = u * texture.width - 0.5f;
    const float y = v * texture.height - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - x0;
    const float ty = y - y0;
    const glm::vec4 lower = glm::mix(texelAt(x0, y0), texelAt(x0 + 1, y0), tx);
    const glm::vec4 upper = glm::mix(texelAt(x0, y0 + 1), texelAt(x0 + 1, y0 + 1), tx);
    return glm::mix(lower, upper, ty);
}

__device__ float srgbToLinear(float value)
{
    return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
}

__device__ glm::vec4 sampleBaseColorTexture(const Material& material, const glm::vec2& uv,
    const DeviceTextureStore& textureStore)
{
    if (!textureStore.validTextureIndex(material.baseColorTexture)) return glm::vec4(1.0f);
    const TextureInfo& texture = textureStore.textures[material.baseColorTexture];
    const glm::vec4 sampled = sampleTexture(texture, uv, textureStore);
    return glm::vec4(srgbToLinear(sampled.r), srgbToLinear(sampled.g),
        srgbToLinear(sampled.b), sampled.a);
}

__device__ glm::vec3 sampleNormalTexture(const Material& material, const TriangleSoA& triangles,
    int triangleIndex,
    const glm::vec2& uv, glm::vec3 geometricNormal, const DeviceTextureStore& textureStore)
{
    const TriangleAttributes& triangle = triangles.attributes[triangleIndex];
    if (!textureStore.validTextureIndex(material.normalTexture) || !triangle.hasTextureCoordinates)
        return geometricNormal;

    const TextureInfo& texture = textureStore.textures[material.normalTexture];
    const glm::vec3 encodedNormal = glm::vec3(sampleTexture(texture, uv, textureStore));
    glm::vec3 tangentSpaceNormal = 2.0f * encodedNormal - glm::vec3(1.0f);
    tangentSpaceNormal.x *= material.normalScale;
    tangentSpaceNormal.y *= material.normalScale;
    if (glm::length2(tangentSpaceNormal) <= EPSILON) return geometricNormal;
    tangentSpaceNormal = glm::normalize(tangentSpaceNormal);

    const glm::vec3& edge1 = triangles.edges1[triangleIndex];
    const glm::vec3& edge2 = triangles.edges2[triangleIndex];
    const glm::vec2 uv1 = triangle.triangleUVs[1] - triangle.triangleUVs[0];
    const glm::vec2 uv2 = triangle.triangleUVs[2] - triangle.triangleUVs[0];
    const float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
    if (fabsf(determinant) <= EPSILON) return geometricNormal;

    const glm::vec3 normal = glm::normalize(geometricNormal);
    glm::vec3 tangent = (uv2.y * edge1 - uv1.y * edge2) / determinant;
    tangent -= normal * glm::dot(normal, tangent);
    if (glm::length2(tangent) <= EPSILON) return geometricNormal;
    tangent = glm::normalize(tangent);

    const glm::vec3 uvBitangent = (uv1.x * edge2 - uv2.x * edge1) / determinant;
    glm::vec3 bitangent = glm::cross(normal, tangent);
    if (glm::dot(bitangent, uvBitangent) < 0.0f) bitangent = -bitangent;
    return glm::normalize(tangentSpaceNormal.x * tangent + tangentSpaceNormal.y * bitangent +
        tangentSpaceNormal.z * normal);
}

__device__ float trowbridgeReitzDistribution(const glm::vec3& wh, float roughness)
{
    const float alpha = glm::clamp(roughness, 0.001f, 1.0f);
    const float alpha2 = alpha * alpha;
    const float cosThetaH = fabsf(wh.z);
    const float denominator = cosThetaH * cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denominator * denominator);
}

__device__ float roughSpecularPdf(const glm::vec3& incoming, const glm::vec3& outgoing,
    glm::vec3 normal, float roughness)
{
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f) normal = -normal;
    const glm::vec3 wo = bsdfWorldToLocal(normal, -glm::normalize(incoming));
    const glm::vec3 wi = bsdfWorldToLocal(normal, glm::normalize(outgoing));
    if (wo.z <= 0.0f || wi.z <= 0.0f || glm::length2(wo + wi) <= EPSILON) return 0.0f;
    const glm::vec3 wh = glm::normalize(wo + wi);
    const float woDotWh = fabsf(glm::dot(wo, wh));
    return woDotWh > EPSILON ? trowbridgeReitzDistribution(wh, roughness) * fabsf(wh.z) /
        (4.0f * woDotWh) : 0.0f;
}

__device__ glm::vec3 evaluateDirectBSDF(const Material& material, const glm::vec3& incoming,
    const glm::vec3& outgoing, glm::vec3 normal)
{
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f) normal = -normal;
    const glm::vec3 wo = bsdfWorldToLocal(normal, -glm::normalize(incoming));
    const glm::vec3 wi = bsdfWorldToLocal(normal, glm::normalize(outgoing));
    if (wo.z <= 0.0f || wi.z <= 0.0f) return glm::vec3(0.0f);

    if (material.type == MATERIAL_DIFFUSE) return material.color / PI;
    if (material.type != MATERIAL_COOK_TORRANCE && material.type != MATERIAL_MICROFACETS)
        return glm::vec3(0.0f);

    const float roughness = glm::clamp(material.roughness, 0.001f, 1.0f);
    const glm::vec3 wh = glm::normalize(wo + wi);
    const float G = 1.0f / (1.0f + bsdfTrowbridgeReitzLambda(wo, roughness) +
        bsdfTrowbridgeReitzLambda(wi, roughness));
    // MICROFACETS uses albedo with unit Fresnel.
    const glm::vec3 f0 = material.type == MATERIAL_MICROFACETS ? material.color :
        glm::mix(glm::vec3(0.04f), material.color,
            glm::clamp(material.metallic, 0.0f, 1.0f));
    glm::vec3 result = f0 * (trowbridgeReitzDistribution(wh, roughness) * G /
        (4.0f * wo.z * wi.z));
    if (material.type == MATERIAL_COOK_TORRANCE)
        result += (1.0f - glm::clamp(material.metallic, 0.0f, 1.0f)) * material.color / PI;
    return result;
}

__device__ float bsdfPdf(const Material& material, const glm::vec3& incoming,
    const glm::vec3& outgoing, glm::vec3 normal)
{
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f) normal = -normal;
    const float diffusePdf = fmaxf(0.0f, glm::dot(glm::normalize(outgoing), normal)) / PI;
    if (material.type == MATERIAL_DIFFUSE) return diffusePdf;
    if (material.type == MATERIAL_MICROFACETS)
        return roughSpecularPdf(incoming, outgoing, normal, material.roughness);
    if (material.type != MATERIAL_COOK_TORRANCE) return 0.0f;
    const glm::vec3 f0 = glm::mix(glm::vec3(0.04f), material.color,
        glm::clamp(material.metallic, 0.0f, 1.0f));
    const float specularProbability = glm::clamp(fmaxf(f0.x, fmaxf(f0.y, f0.z)), 0.05f, 0.95f);
    return specularProbability * roughSpecularPdf(incoming, outgoing, normal, material.roughness) +
        (1.0f - specularProbability) * diffusePdf;
}

// Sample one direct-light strategy and enqueue its visibility ray.  The
// contribution is applied later only when traceShadowRays marks it visible.
__device__ void enqueueDirectLighting(const Material& material, const PathSegment& pathSegment,
    const glm::vec3& intersectionPoint, const glm::vec3& normal, const DevicePrimitiveStore& primitiveStore,
    const Material* materials, const int* lightPrimitives, int emissiveLightCount,
    const DeviceEnvironmentMap& environment, int totalLightCount, int pathIndex,
    ShadowRay& shadowRay, thrust::default_random_engine& rng)
{
    shadowRay.active = 0;
    shadowRay.pathIndex = pathIndex;
    if (totalLightCount <= 0) return;
    thrust::uniform_int_distribution<int> chooseLight(0, totalLightCount - 1);
    const int chosenLight = chooseLight(rng);
    const glm::vec3 offsetNormal = glm::dot(pathSegment.ray.direction, normal) < 0.0f ? normal : -normal;

    // Environment is the sole non-geometric light entry.
    if (chosenLight == emissiveLightCount)
    {
        const EnvironmentSample sample = sampleEnvironment(environment, rng);
        if (!sample.valid) return;
        const float cosSurface = fmaxf(0.0f, glm::dot(offsetNormal, sample.direction));
        if (cosSurface <= 0.0f) return;
        const float lightPdf = sample.pdfSolidAngle / static_cast<float>(totalLightCount);
        const float scatteringPdf = bsdfPdf(material, pathSegment.ray.direction, sample.direction, normal);
        if (lightPdf <= EPSILON) return;
        const float lightPdf2 = lightPdf * lightPdf;
        const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
        shadowRay.ray = { intersectionPoint + 0.001f * offsetNormal, sample.direction };
        shadowRay.maxDistance = FLT_MAX;
        shadowRay.contribution = pathSegment.color *
            evaluateDirectBSDF(material, pathSegment.ray.direction, sample.direction, normal) *
            sample.radiance * (cosSurface * misWeight / lightPdf);
        shadowRay.occluded = 0;
        shadowRay.active = 1;
        return;
    }

    if (chosenLight >= emissiveLightCount) return;
    const PrimitiveRef& light = primitiveStore.primitives[lightPrimitives[chosenLight]];
    const LightSurfaceSample sample = sampleLightSurface(light, primitiveStore, rng);
    if (!sample.valid) return;

    const glm::vec3 shadowOrigin = intersectionPoint + 0.001f * offsetNormal;
    const glm::vec3 toLight = sample.position - shadowOrigin;
    const float distance2 = glm::dot(toLight, toLight);
    if (distance2 <= EPSILON) return;
    const float distance = sqrtf(distance2);
    const glm::vec3 wi = toLight / distance;
    const float cosSurface = fmaxf(0.0f, glm::dot(offsetNormal, wi));
    // Emissive primitives radiate from their front side only.  Their front
    // side is the outward primitive normal.
    const float cosLight = glm::dot(sample.normal, -wi);
    if (cosSurface <= 0.0f || cosLight <= EPSILON) return;

    const float lightPdf = distance2 * sample.pdfArea /
        (cosLight * static_cast<float>(totalLightCount));
    const float scatteringPdf = bsdfPdf(material, pathSegment.ray.direction, wi, normal);
    if (lightPdf <= EPSILON) return;
    const float lightPdf2 = lightPdf * lightPdf;
    const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
    const Material& lightMaterial = materials[primitiveMaterialId(light, primitiveStore)];
    shadowRay.ray = { shadowOrigin, wi };
    shadowRay.maxDistance = distance - 0.001f;
    shadowRay.contribution = pathSegment.color *
        evaluateDirectBSDF(material, pathSegment.ray.direction, wi, normal) *
        emittedRadiance(lightMaterial) * (cosSurface * misWeight / lightPdf);
    shadowRay.occluded = 0;
    shadowRay.active = 1;
}

__global__ void shadeBSDF(
    int iter,
    int depth,
    int traceDepth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    ShadowRay* shadowRays,
    Material* materials,
    DevicePrimitiveStore primitiveStore,
    const int* lightPrimitives,
    int emissiveLightCount,
    DeviceEnvironmentMap environment,
    DeviceTextureStore textureStore,
    int lightCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        // The queue is reused for every bounce; paths that do not sample a
        // non-delta direct-light strategy leave an inactive entry.
        shadowRays[idx].active = 0;
        const ShadeableIntersection& intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            PathSegment pathSegment = pathSegments[idx];
            const PrimitiveRef& hitPrimitive = primitiveStore.primitives[intersection.primitiveIndex];
            Material material = materials[primitiveMaterialId(hitPrimitive, primitiveStore)];
            glm::vec2 surfaceUV(0.0f);
            if (hitPrimitive.type == TRIANGLE)
            {
                const TriangleAttributes& attributes = primitiveStore.triangles.attributes[hitPrimitive.index];
                if (attributes.hasTextureCoordinates)
                {
                    const float u = intersection.barycentrics.x;
                    const float v = intersection.barycentrics.y;
                    surfaceUV = (1.0f - u - v) * attributes.triangleUVs[0] +
                        u * attributes.triangleUVs[1] + v * attributes.triangleUVs[2];
                }
            }

            // Sample texture for color and alpha
            glm::vec4 baseColorSample(1.0f);
            if (hitPrimitive.type == TRIANGLE &&
                primitiveStore.triangles.attributes[hitPrimitive.index].hasTextureCoordinates)
            {
                baseColorSample = sampleBaseColorTexture(material, surfaceUV, textureStore);
                material.color *= glm::vec3(baseColorSample);
            }

            const float opacity = glm::clamp(material.alpha * baseColorSample.a, 0.0f, 1.0f);
            bool ignoreIntersection = false;
            if (material.alphaMode == ALPHA_MASK)
            {
                ignoreIntersection = opacity < material.alphaCutoff;
            }
            else if (material.alphaMode == ALPHA_BLEND)
            {
                // Stochastically selecting the surface or the ray behind it
                // gives the expected alpha blend without splitting the path.
                thrust::default_random_engine alphaRng = makeSeededRandomEngine(
                    iter, pathSegment.pixelIndex, depth);
                thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
                ignoreIntersection = u01(alphaRng) >= opacity;
            }

            if (ignoreIntersection)
            {
                // Alpha transparency: ignore the hit, keep the throughput, BSDF history
                const glm::vec3 intersectionPoint = pathSegment.ray.origin +
                    intersection.t * pathSegment.ray.direction;
                pathSegment.ray.origin = intersectionPoint + 0.001f * pathSegment.ray.direction;
                pathSegments[idx] = pathSegment;
                return;
            }
            glm::vec3 emission = material.emission;
            if (fmaxf(emission.x, fmaxf(emission.y, emission.z)) <= 0.0f && material.emittance > 0.0f)
            {
                emission = material.color * material.emittance;
            }
            if (material.type == MATERIAL_EMISSIVE || fmaxf(emission.x, fmaxf(emission.y, emission.z)) > 0.0f) {
                const glm::vec3 lightPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                const glm::vec3 hitLightNormal = lightSurfaceNormal(hitPrimitive, primitiveStore, lightPoint);
                const bool frontFacing = glm::dot(hitLightNormal,
                    -glm::normalize(pathSegment.ray.direction)) > EPSILON;
                if (frontFacing)
                {
                    float misWeight = 1.0f;
                    if (!pathSegment.previousBounceWasSpecular && pathSegment.previousBsdfPdf > 0.0f)
                    {
                        const float lightPdf = lightPdfSolidAngle(hitPrimitive, primitiveStore,
                            pathSegment.ray.origin, lightPoint, hitLightNormal, lightCount);
                        const float bsdfPdf2 = pathSegment.previousBsdfPdf * pathSegment.previousBsdfPdf;
                        misWeight = bsdfPdf2 / (bsdfPdf2 + lightPdf * lightPdf);
                    }
                    pathSegment.radiance += pathSegment.color * emission * misWeight;
                }
                pathSegment.remainingBounces = 0;
            }
            else {
                glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
                const glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                glm::vec3 geometricNormal = lightSurfaceNormal(hitPrimitive, primitiveStore, hitPoint);
                if (hitPrimitive.type == TRIANGLE)
                {
                    const TriangleAttributes& attributes = primitiveStore.triangles.attributes[hitPrimitive.index];
                    if (attributes.hasVertexNormals)
                    {
                        const float u = intersection.barycentrics.x;
                        const float v = intersection.barycentrics.y;
                        const glm::vec3 normal = (1.0f - u - v) * attributes.triangleNormals[0] +
                            u * attributes.triangleNormals[1] + v * attributes.triangleNormals[2];
                        if (glm::dot(normal, normal) >= 1e-16f) geometricNormal = glm::normalize(normal);
                    }
                }
                if (hitPrimitive.type == TRIANGLE &&
                    primitiveStore.triangles.attributes[hitPrimitive.index].hasTextureCoordinates)
                {
                    geometricNormal = sampleNormalTexture(material, primitiveStore.triangles, hitPrimitive.index, surfaceUV,
                        geometricNormal, textureStore);
                }
                bool enteringDielectric = glm::dot(incoming, geometricNormal) < 0.0f;
                glm::vec3 normal = geometricNormal;
                if (!enteringDielectric)
                {
                    normal = -normal;
                }
                thrust::default_random_engine rng = makeSeededRandomEngine(
                    iter,
                    pathSegment.pixelIndex,
                    pathSegment.remainingBounces);
                glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                // Sample emissive primitive at non-delta vertex
				// Use power heuristic to weight the BSDF and light PDFs
                if (material.type != MATERIAL_MIRROR && material.type != MATERIAL_DIELECTRIC)
                {
                    enqueueDirectLighting(material,
                        pathSegment, intersect, normal, primitiveStore, materials, lightPrimitives, emissiveLightCount,
                        environment, lightCount, idx, shadowRays[idx], rng);
                }
                switch (material.type)
                {
                case MATERIAL_MIRROR:
                    pathSegment.color *= material.color;
                    scatterMirror(pathSegment, intersect, normal);
                    pathSegment.previousBsdfPdf = 0.0f;
                    pathSegment.previousBounceWasSpecular = true;
                    break;
                case MATERIAL_DIELECTRIC:
                    scatterDielectric(pathSegment, intersect, geometricNormal, material, rng);
                    if (glm::dot(pathSegment.ray.direction, normal) < 0.0f)
                    {
                        float eta = material.indexOfRefraction > 0.0f ? material.indexOfRefraction : 1.55f;
                        float etaRatio = enteringDielectric ? 1.0f / eta : eta;
                        pathSegment.color *= material.color * (etaRatio * etaRatio);
                    }
                    pathSegment.previousBsdfPdf = 0.0f;
                    pathSegment.previousBounceWasSpecular = true;
                    break;
                case MATERIAL_COOK_TORRANCE:
                {
                    // glTF's metallic-roughness model contains both a diffuse
                    // dielectric lobe and a GGX specular lobe.  Choose one
                    // lobe per bounce and compensate for that choice in the
                    // throughput so the result remains an unbiased mixture.
                    const float roughness = glm::clamp(material.roughness, 0.001f, 1.0f);
                    const float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
                    const glm::vec3 f0 = glm::mix(glm::vec3(0.04f), material.color, metallic);
                    const float specularProbability = glm::clamp(
                        fmaxf(f0.x, fmaxf(f0.y, f0.z)), 0.05f, 0.95f);
                    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
                    if (u01(rng) < specularProbability)
                    {
                        scatterRoughSpecular(pathSegment, intersect, normal, roughness, rng);
                        pathSegment.color *= roughSpecularThroughput(
                            incoming, pathSegment.ray.direction, normal, material, roughness) /
                            specularProbability;
                        pathSegment.previousBsdfPdf = bsdfPdf(material, incoming,
                            pathSegment.ray.direction, normal);
                        pathSegment.previousBounceWasSpecular = false;
                    }
                    else
                    {
                        pathSegment.color *= ((1.0f - metallic) * material.color) /
                            (1.0f - specularProbability);
                        scatterRay(pathSegment, intersect, normal, rng);
                        pathSegment.previousBsdfPdf = bsdfPdf(material, incoming,
                            pathSegment.ray.direction, normal);
                        pathSegment.previousBounceWasSpecular = false;
                    }
                    break;
                }
                case MATERIAL_MICROFACETS:
                    if (scatterRoughSpecular(pathSegment, intersect, normal, material.roughness, rng))
                    {
                        pathSegment.color *= roughSpecularThroughput(
                            incoming, pathSegment.ray.direction, normal, material, material.roughness);
                        pathSegment.previousBsdfPdf = bsdfPdf(material, incoming,
                            pathSegment.ray.direction, normal);
                    }
                    else
                    {
                        // Sample_f_microfacet_refl returns black when the
                        // reflected direction leaves wo's hemisphere.
                        pathSegment.color = glm::vec3(0.0f);
                        pathSegment.previousBsdfPdf = 0.0f;
                        pathSegment.remainingBounces = 0;
                    }
                    pathSegment.previousBounceWasSpecular = false;
                    break;
                case MATERIAL_DIFFUSE:
                default:
                    pathSegment.color *= material.color;
                    scatterRay(pathSegment, intersect, geometricNormal, rng);
                    pathSegment.previousBsdfPdf = bsdfPdf(material, incoming,
                        pathSegment.ray.direction, geometricNormal);
                    pathSegment.previousBounceWasSpecular = false;
                    break;
                }
                --pathSegment.remainingBounces;

#if RUSSIAN_ROULETTE
                const int completedBounces = traceDepth - pathSegment.remainingBounces;
                if (completedBounces >= RUSSIAN_ROULETTE_START_DEPTH && pathSegment.remainingBounces > 0)
                {
                    const float survivalProbability = glm::clamp(
                        fmaxf(pathSegment.color.x, fmaxf(pathSegment.color.y, pathSegment.color.z)),
                        0.05f,
                        0.95f);
                    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
                    if (u01(rng) > survivalProbability)
                    {
                        pathSegment.color = glm::vec3(0.0f);
                        pathSegment.remainingBounces = 0;
                    }
                    else
                    {
                        pathSegment.color /= survivalProbability;
                    }
                }
#endif
            }
            pathSegments[idx] = pathSegment;
        }
        else {
			// Ray hit nothing, so sample the environment map if present
            PathSegment pathSegment = pathSegments[idx];
            if (environment.valid())
            {
                float misWeight = 1.0f;
                if (!pathSegment.previousBounceWasSpecular && pathSegment.previousBsdfPdf > 0.0f)
                {
                    const float lightPdf = environmentPdf(environment, pathSegment.ray.direction) /
                        static_cast<float>(lightCount);
                    const float bsdfPdf2 = pathSegment.previousBsdfPdf * pathSegment.previousBsdfPdf;
                    misWeight = bsdfPdf2 / (bsdfPdf2 + lightPdf * lightPdf);
                }
                pathSegment.radiance += pathSegment.color *
                    environmentRadiance(environment, pathSegment.ray.direction) * misWeight;
            }
            pathSegment.remainingBounces = 0;
            pathSegments[idx] = pathSegment;
		}
    }
}

// Traverse only the visibility rays enqueued by shadeBSDF.
__global__ void traceShadowRays(
    int numShadowRays,
    ShadowRay* shadowRays,
    const int* rayOrder)
{
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= numShadowRays) return;
    const int index = rayOrder ? rayOrder[lane] : lane;

    const Ray ray = shadowRays[index].ray;
    shadowRays[index].occluded = isOccluded(ray, shadowRays[index].maxDistance) ? 1 : 0;
}

// A path owns at most one queue entry per bounce, so these writes are
// contention-free and can be applied before terminated paths are gathered.
__global__ void accumulateVisibleDirectLighting(
    int numShadowRays,
    const ShadowRay* shadowRays,
    PathSegment* pathSegments)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numShadowRays) return;

    const ShadowRay& shadowRay = shadowRays[index];
    if (shadowRay.active && !shadowRay.occluded)
    {
        pathSegments[shadowRay.pathIndex].radiance += shadowRay.contribution;
    }
}

__global__ void buildMaterialSortKeys(
    int num_paths,
    int material_count,
    ShadeableIntersection* shadeableIntersections,
    Material* materials,
    DevicePrimitiveStore primitiveStore,
    int* materialSortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        const ShadeableIntersection& intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            const int materialId = primitiveMaterialId(
                primitiveStore.primitives[intersection.primitiveIndex], primitiveStore);
            materialSortKeys[idx] = static_cast<int>(materials[materialId].type) * material_count + materialId;
        }
        else
        {
            materialSortKeys[idx] = MATERIAL_TYPE_COUNT * material_count;
        }
    }
}

#if MORTON_SORT
__device__ __forceinline__ unsigned int spreadMortonBits(unsigned int value)
{
    value = (value | (value << 16)) & 0x030000ffu;
    value = (value | (value << 8)) & 0x0300f00fu;
    value = (value | (value << 4)) & 0x030c30c3u;
    return (value | (value << 2)) & 0x09249249u;
}

__device__ __forceinline__ unsigned int mortonCode(const glm::vec3& value, float bins)
{
    const unsigned int x = static_cast<unsigned int>(fminf(bins - 1.0f, fmaxf(0.0f, value.x * bins)));
    const unsigned int y = static_cast<unsigned int>(fminf(bins - 1.0f, fmaxf(0.0f, value.y * bins)));
    const unsigned int z = static_cast<unsigned int>(fminf(bins - 1.0f, fmaxf(0.0f, value.z * bins)));
    return spreadMortonBits(x) | (spreadMortonBits(y) << 1) | (spreadMortonBits(z) << 2);
}

// 18 origin bits (64 cells/axis) followed by 12 direction bits (16 bins/axis).
// Camera rays sharing an origin are consequently ordered by direction too.
// Sort indices only: paths, hits, pixel seeds, and shadow ownership stay paired.
template <typename RayPayload>
__global__ void buildMortonSortKeys(int count, const RayPayload* rays,
    glm::vec3 boundsMin, glm::vec3 inverseExtent, unsigned int* keys, int* order)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const Ray& ray = rays[index].ray;
    const unsigned int origin = mortonCode((ray.origin - boundsMin) * inverseExtent, 64.0f);
    const unsigned int direction = mortonCode(0.5f * ray.direction + glm::vec3(0.5f), 16.0f);
    keys[index] = (origin << 12) | direction;
    order[index] = index;
}

template <typename RayPayload>
void orderRays(int count, const RayPayload* rays, int blockSize)
{
    buildMortonSortKeys<<<(count + blockSize - 1) / blockSize, blockSize>>>(
        count, rays, mortonBoundsMin, mortonInverseExtent, dev_mortonSortKeys, dev_rayOrder);
    thrust::sort_by_key(thrust::device, dev_mortonSortKeys, dev_mortonSortKeys + count, dev_rayOrder);
    checkCUDAError("sort traversal rays by Morton code");
}
#endif

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.radiance;
    }
}

// Add paths that are about to be compacted away to the accumulated image.
__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces <= 0)
        {
            image[iterationPath.pixelIndex] += iterationPath.radiance;
        }
    }
}

struct IsTerminated {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return p.remainingBounces <= 0;
    }
};

struct IsInactiveShadowRay {
    __host__ __device__
        bool operator()(const ShadowRay& s) const {
        return !s.active;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
    const int totalLightCount = dev_lightPrimitiveCount + (dev_environment.valid() ? 1 : 0);

    const float focalLength = guiData ? guiData->FocalLength : 2.0f;
    const float lensRadius = guiData ? guiData->LensRadius : 0.008f;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, focalLength, lensRadius);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = num_paths == 0;
    while (!iterationComplete)
    {
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if MORTON_SORT
        orderRays(num_paths, dev_paths, blockSize1d);
#endif
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            num_paths,
            dev_paths,
            dev_rayOrder,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if MATERIAL_SORT
        buildMaterialSortKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            hst_scene->materials.size(),
            dev_intersections,
            dev_materials,
            dev_primitiveStore,
            dev_materialSortKeys
        );
        checkCUDAError("build material sort keys");

        auto sortedValues = thrust::make_zip_iterator(thrust::make_tuple(dev_intersections, dev_paths));
        thrust::sort_by_key(
            thrust::device,
            dev_materialSortKeys,
            dev_materialSortKeys + num_paths,
            sortedValues
        );
        checkCUDAError("sort paths by material type");

#endif
        shadeBSDF<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            traceDepth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_shadowRays,
            dev_materials,
            dev_primitiveStore,
            dev_lightPrimitives,
            dev_lightPrimitiveCount,
            dev_environment,
            dev_textureStore,
            totalLightCount
        );
        checkCUDAError("shade materials and enqueue shadow rays");

        // Compact the queue before traversal. pathIndex remains valid after
        // the move, so visible contributions are still written to their owner.
        ShadowRay* dev_shadow_end = thrust::remove_if(
            thrust::device,
            dev_shadowRays,
            dev_shadowRays + num_paths,
            IsInactiveShadowRay());
        const int numShadowRays = static_cast<int>(dev_shadow_end - dev_shadowRays);

        if (numShadowRays > 0)
        {
            const dim3 shadowBlocks = (numShadowRays + blockSize1d - 1) / blockSize1d;
#if MORTON_SORT
            orderRays(numShadowRays, dev_shadowRays, blockSize1d);
#endif
            traceShadowRays<<<shadowBlocks, blockSize1d>>>(
                numShadowRays,
                dev_shadowRays,
                dev_rayOrder
            );
            checkCUDAError("trace shadow rays");

            accumulateVisibleDirectLighting<<<shadowBlocks, blockSize1d>>>(
                numShadowRays,
                dev_shadowRays,
                dev_paths
            );
            checkCUDAError("accumulate visible direct lighting");
        }

        gatherTerminatedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_image,
            dev_paths
        );
        checkCUDAError("gather terminated paths");

		//stream compaction: remove terminated paths from dev_paths
        dev_path_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsTerminated());
        num_paths = dev_path_end - dev_paths;

		// iteration can be greater than traceDepth if some paths are still active due to alpha blending
		iterationComplete = num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Completed paths have already been accumulated before compaction.

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    checkCUDAError("pathtrace");
}

void pathtraceCopyImageToHost()
{
    if (hst_scene == NULL || dev_image == NULL) return;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("copy rendered image to host");
}

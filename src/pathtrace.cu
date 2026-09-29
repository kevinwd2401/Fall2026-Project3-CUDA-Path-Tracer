#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cstring>
#include <utility>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

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
// Avoid radix/key-generation launches for small tail queues. This is a tuning
// heuristic; set it to 0 to compare against sorting every queue.
#ifndef MORTON_SORT_MIN_RAYS
#define MORTON_SORT_MIN_RAYS 4096
#endif
#ifndef TRIANGLE_TRAVERSAL
#define TRIANGLE_TRAVERSAL 1
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
// A leaf always escapes to the next node in this depth-first BVH. Reuse its
// escape slot for the primitive count, leaving just two metadata words/node.
struct TraversalNode
{
    float4 lower; // xyz: minimum, w bits: first primitive, or -1 for an interior
    float4 upper; // xyz: maximum, w bits: leaf count or interior escape index
};
static_assert(sizeof(TraversalNode) == 32, "Traversal nodes must occupy 32 bytes");

struct TraversalTriangle
{
    float4 vertex; // w bits: original scene primitive ID (for shading)
    float4 edge1;
    float4 edge2;
};
static_assert(sizeof(TraversalTriangle) == 48, "Traversal triangles must occupy 48 bytes");

static TraversalNode* dev_bvhNodes = NULL;
static TraversalTriangle* dev_traversalTriangles = NULL;
static bool trianglesOnly = false;
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

// Uniform, read-only descriptors live in constant memory. Triangle-only
// scenes bypass primitive dispatch and the two dependent leaf index lookups.
struct TraversalData
{
    const PrimitiveRef* primitives;
    const Cube* cubes;
    const Sphere* spheres;
    const glm::vec3* vertices;
    const glm::vec3* edges1;
    const glm::vec3* edges2;
    const TraversalNode* nodes;
    const TraversalTriangle* orderedTriangles;
    const int* primitiveIndices;
    int nodeCount;
};

__constant__ TraversalData traversalData;

static glm::vec3 mortonBoundsMin(0.0f);
static glm::vec3 mortonInverseExtent(0.0f);

// Bit-copy metadata rather than converting it to float (large IDs must retain
// every bit). Device code decodes w with __float_as_int.
float4 packTraversalData(const glm::vec3& value, int metadata)
{
    float4 packed = make_float4(value.x, value.y, value.z, 0.0f);
    std::memcpy(&packed.w, &metadata, sizeof(metadata));
    return packed;
}

// A shading thread owns the slot at its current path index. Only active slot
// indices are compacted/sorted; the payload stays in place until traversal ends.
struct ShadowRay
{
    Ray ray;
    glm::vec3 contribution;
    float maxDistance;
    int pathIndex;
    int active;
};

static PathSegment* dev_paths = NULL;
static PathSegment* dev_pathsAlternate = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_intersectionsAlternate = NULL;
static ShadowRay* dev_shadowRays = NULL;
// Only the textured surface values cross shading stages. A negative material
// ID marks misses, emitters, and alpha pass-throughs. Paths/hits keep their
// indices until direct lighting has been accumulated and paths are compacted.
struct ShadingState
{
    glm::vec3 normal;
    glm::vec3 color;
    int materialId;
    thrust::default_random_engine rng;
};

static ShadingState* dev_shadingStates = NULL;
// All selection and radix-sort operations run sequentially on the default
// stream and share one allocation sized for the largest operation at init.
static void* dev_queueScratch = NULL;
static size_t queueScratchBytes = 0;
static int* dev_selectedCount = NULL;
static unsigned int* dev_sortKeys[2] = { NULL, NULL };
static int* dev_sortIndices[2] = { NULL, NULL };
static int materialSortBits = 1;
static constexpr int mortonSortBits = 30;

struct IsActivePath
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};

struct IsActiveShadowIndex
{
    const ShadowRay* rays;

    __device__ bool operator()(int index) const
    {
        return rays[index].active != 0;
    }
};

#if MATERIAL_SORT || MORTON_SORT
size_t radixSortScratchSize(int count, int endBit)
{
    cub::DoubleBuffer<unsigned int> keys(dev_sortKeys[0], dev_sortKeys[1]);
    cub::DoubleBuffer<int> indices(dev_sortIndices[0], dev_sortIndices[1]);
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, bytes, keys, indices, count, 0, endBit);
    return bytes;
}

const int* sortRayIndices(int count, int endBit)
{
    if (count < 2) return dev_sortIndices[0];
    // Each key-generation pass starts in buffer 0. CUB chooses the final
    // buffer according to the radix pass count; never assume it is buffer 1.
    cub::DoubleBuffer<unsigned int> keys(dev_sortKeys[0], dev_sortKeys[1]);
    cub::DoubleBuffer<int> indices(dev_sortIndices[0], dev_sortIndices[1]);
    size_t bytes = queueScratchBytes;
    cub::DeviceRadixSort::SortPairs(dev_queueScratch, bytes, keys, indices, count, 0, endBit);
    return indices.Current();
}
#endif

void initQueueOperations(int capacity, int materialCount)
{
    cudaMalloc(&dev_pathsAlternate, capacity * sizeof(PathSegment));
    cudaMalloc(&dev_selectedCount, sizeof(int));
    // Buffer 0 is also the compacted list of active shadow indices, including
    // builds with both sorting options disabled.
    cudaMalloc(&dev_sortIndices[0], capacity * sizeof(int));
#if MATERIAL_SORT || MORTON_SORT
    cudaMalloc(&dev_sortIndices[1], capacity * sizeof(int));
    cudaMalloc(&dev_sortKeys[0], capacity * sizeof(unsigned int));
    cudaMalloc(&dev_sortKeys[1], capacity * sizeof(unsigned int));
#endif
#if MATERIAL_SORT
    cudaMalloc(&dev_intersectionsAlternate, capacity * sizeof(ShadeableIntersection));
    // Include the miss sentinel, which is greater than every material key.
    unsigned int maximumKey = static_cast<unsigned int>(MATERIAL_TYPE_COUNT) *
        static_cast<unsigned int>(materialCount);
    materialSortBits = 0;
    do
    {
        ++materialSortBits;
        maximumKey >>= 1;
    } while (maximumKey != 0);
#endif

    size_t bytes = 0;
    cub::DeviceSelect::If(NULL, bytes, dev_paths, dev_pathsAlternate,
        dev_selectedCount, capacity, IsActivePath());
    queueScratchBytes = bytes;
    bytes = 0;
    cub::DeviceSelect::If(NULL, bytes, thrust::counting_iterator<int>(0),
        dev_sortIndices[0], dev_selectedCount, capacity, IsActiveShadowIndex{ dev_shadowRays });
    if (bytes > queueScratchBytes) queueScratchBytes = bytes;
#if MORTON_SORT
    bytes = radixSortScratchSize(capacity, mortonSortBits);
    if (bytes > queueScratchBytes) queueScratchBytes = bytes;
#endif
#if MATERIAL_SORT
    bytes = radixSortScratchSize(capacity, materialSortBits);
    if (bytes > queueScratchBytes) queueScratchBytes = bytes;
#endif
    cudaMalloc(&dev_queueScratch, queueScratchBytes);
}

int compactShadowIndices(int count)
{
    size_t bytes = queueScratchBytes;
    cub::DeviceSelect::If(dev_queueScratch, bytes, thrust::counting_iterator<int>(0),
        dev_sortIndices[0], dev_selectedCount, count, IsActiveShadowIndex{ dev_shadowRays });
    int selected = 0;
    // The host still needs the count for radix sorting and launch dimensions.
    cudaMemcpy(&selected, dev_selectedCount, sizeof(int), cudaMemcpyDeviceToHost);
    return selected;
}

int compactPaths(int count)
{
    size_t bytes = queueScratchBytes;
    cub::DeviceSelect::If(dev_queueScratch, bytes, dev_paths, dev_pathsAlternate,
        dev_selectedCount, count, IsActivePath());
    int selected = 0;
    cudaMemcpy(&selected, dev_selectedCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::swap(dev_paths, dev_pathsAlternate);
    return selected;
}

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

    trianglesOnly = TRIANGLE_TRAVERSAL && !scene->primitives.empty();
    for (const PrimitiveRef& primitive : scene->primitives)
        trianglesOnly = trianglesOnly && primitive.type == TRIANGLE;
    if (trianglesOnly)
    {
        std::vector<TraversalTriangle> orderedTriangles;
        orderedTriangles.reserve(scene->bvhPrimitiveIndices.size());
        for (int primitiveIndex : scene->bvhPrimitiveIndices)
        {
            const Triangle& triangle = scene->triangles[scene->primitives[primitiveIndex].index];
            orderedTriangles.push_back({
                packTraversalData(triangle.triangleVertices[0], primitiveIndex),
                packTraversalData(triangle.triangleVertices[1] - triangle.triangleVertices[0], 0),
                packTraversalData(triangle.triangleVertices[2] - triangle.triangleVertices[0], 0) });
        }
        dev_traversalTriangles = uploadArray(orderedTriangles);
    }

    mortonBoundsMin = glm::vec3(0.0f);
    mortonInverseExtent = glm::vec3(0.0f);
    if (!scene->bvhNodes.empty())
    {
        std::vector<TraversalNode> nodes;
        nodes.reserve(scene->bvhNodes.size());
        for (const BVHNode& node : scene->bvhNodes)
        {
            const bool leaf = node.primitiveCount > 0;
            nodes.push_back({ packTraversalData(node.boundsMin, leaf ? node.firstPrimitive : -1),
                packTraversalData(node.boundsMax, leaf ? node.primitiveCount : node.escapeIndex) });
        }
        dev_bvhNodes = uploadArray(nodes);
        if (!trianglesOnly) dev_bvhPrimitiveIndices = uploadArray(scene->bvhPrimitiveIndices);
        mortonBoundsMin = scene->bvhNodes[0].boundsMin;
        const glm::vec3 extent = scene->bvhNodes[0].boundsMax - mortonBoundsMin;
        for (int axis = 0; axis < 3; ++axis)
            mortonInverseExtent[axis] = extent[axis] > 1e-8f ? 1.0f / extent[axis] : 0.0f;
    }
    const TraversalData traversal = { dev_primitives, dev_cubes, dev_spheres,
        dev_triangleVertices, dev_triangleEdges1, dev_triangleEdges2,
        dev_bvhNodes, dev_traversalTriangles, dev_bvhPrimitiveIndices,
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
    cudaMalloc(&dev_shadingStates, pixelcount * sizeof(ShadingState));

    initQueueOperations(pixelcount, static_cast<int>(scene->materials.size()));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_pathsAlternate);
    cudaFree(dev_primitives);
    cudaFree(dev_cubes);
    cudaFree(dev_spheres);
    cudaFree(dev_triangleVertices);
    cudaFree(dev_triangleEdges1);
    cudaFree(dev_triangleEdges2);
    cudaFree(dev_triangleAttributes);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_traversalTriangles);
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
    cudaFree(dev_intersectionsAlternate);
    cudaFree(dev_shadowRays);
    cudaFree(dev_shadingStates);
    cudaFree(dev_queueScratch);
    cudaFree(dev_selectedCount);
    cudaFree(dev_sortKeys[0]);
    cudaFree(dev_sortKeys[1]);
    cudaFree(dev_sortIndices[0]);
    cudaFree(dev_sortIndices[1]);
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
    dev_shadingStates = NULL;
    dev_image = NULL;
    dev_paths = NULL;
    dev_pathsAlternate = NULL;
    dev_intersections = NULL;
    dev_intersectionsAlternate = NULL;
    dev_bvhNodes = NULL;
    dev_traversalTriangles = NULL;
    trianglesOnly = false;
    dev_bvhPrimitiveIndices = NULL;
    dev_materials = NULL;
    dev_lightPrimitives = NULL;
    dev_lightPrimitiveCount = 0;
    dev_queueScratch = NULL;
    queueScratchBytes = 0;
    dev_selectedCount = NULL;
    dev_sortKeys[0] = dev_sortKeys[1] = NULL;
    dev_sortIndices[0] = dev_sortIndices[1] = NULL;
    materialSortBits = 1;

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

template <bool TrianglesOnly>
__device__ __forceinline__ float intersectLeafPrimitive(int slot, const Ray& ray,
    float maxDistance, glm::vec2& barycentrics, int& primitiveIndex)
{
    if constexpr (TrianglesOnly)
    {
        const TraversalTriangle& triangle = traversalData.orderedTriangles[slot];
        const float4 vertex = triangle.vertex;
        const float4 edge1 = triangle.edge1;
        const float4 edge2 = triangle.edge2;
        primitiveIndex = __float_as_int(vertex.w);
        return triangleDistanceTest(glm::vec3(vertex.x, vertex.y, vertex.z),
            glm::vec3(edge1.x, edge1.y, edge1.z), glm::vec3(edge2.x, edge2.y, edge2.z),
            ray, maxDistance, barycentrics);
    }
    else
    {
        primitiveIndex = traversalData.primitiveIndices[slot];
        return intersectPrimitiveDistance(traversalData.primitives[primitiveIndex],
            ray, maxDistance, barycentrics);
    }
}

__device__ __forceinline__ bool intersectTraversalNode(const Ray& ray,
    const glm::vec3& inverseDirection, bool fastBounds, const float4& lower,
    const float4& upper, float maxDistance)
{
    const glm::vec3 minimum(lower.x, lower.y, lower.z);
    const glm::vec3 maximum(upper.x, upper.y, upper.z);
    return fastBounds ? traversalBoundsTestFast(ray, inverseDirection, minimum, maximum, maxDistance) :
        traversalBoundsTest(ray, inverseDirection, minimum, maximum, maxDistance);
}

template <bool TrianglesOnly>
__global__ void computeIntersections(int numPaths, const PathSegment* pathSegments,
    const int* rayOrder, ShadeableIntersection* intersections)
{
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= numPaths) return;
    const int pathIndex = rayOrder ? rayOrder[lane] : lane;
    // Load only the six ray floats, never the full path/throughput payload.
    const Ray ray = pathSegments[pathIndex].ray;
    const glm::vec3 inverseDirection = rayInverseDirection(ray);
    const bool fastBounds = hasFiniteRayReciprocals(inverseDirection);
    float closest = FLT_MAX;
    int hitPrimitive = -1;
    glm::vec2 hitBarycentrics(0.0f);
    int nodeIndex = 0;
    while (nodeIndex < traversalData.nodeCount)
    {
        const float4 lower = traversalData.nodes[nodeIndex].lower;
        const float4 upper = traversalData.nodes[nodeIndex].upper;
        const int firstPrimitive = __float_as_int(lower.w);
        const int countOrEscape = __float_as_int(upper.w);
        if (!intersectTraversalNode(ray, inverseDirection, fastBounds, lower, upper, closest))
        {
            nodeIndex = firstPrimitive < 0 ? countOrEscape : nodeIndex + 1;
            continue;
        }
        if (firstPrimitive < 0)
        {
            ++nodeIndex;
            continue;
        }
        for (int offset = 0; offset < countOrEscape; ++offset)
        {
            int primitiveIndex;
            glm::vec2 barycentrics(0.0f);
            const float t = intersectLeafPrimitive<TrianglesOnly>(firstPrimitive + offset,
                ray, closest, barycentrics, primitiveIndex);
            if (t > 0.0f)
            {
                closest = t;
                hitPrimitive = primitiveIndex;
                hitBarycentrics = barycentrics;
            }
        }
        ++nodeIndex;
    }
    ShadeableIntersection& hit = intersections[pathIndex];
    hit.t = hitPrimitive < 0 ? -1.0f : closest;
    hit.primitiveIndex = hitPrimitive;
    hit.barycentrics = hitBarycentrics;
}

// Isotropic GGX depends only on normal cosines and half-vector dot products.
// Evaluating these in world space avoids building a tangent frame per PDF/BSDF.
__device__ float bsdfTrowbridgeReitzLambda(float cosTheta, float roughness)
{
    if (fabsf(cosTheta) <= 0.0f)
    {
        return 0.0f;
    }

    float absTanTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta)) / fabsf(cosTheta);
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

    const glm::vec3 wo = -incoming;
    const float metallic = glm::clamp(material.metallic, 0.0f, 1.0f);
    // MICROFACET_REFL lobe uses albedo directly with F = 1.
    const glm::vec3 f0 = material.type == MATERIAL_MICROFACETS ? material.color :
        glm::mix(glm::vec3(0.04f), material.color, metallic);
    if (glm::length2(outgoing - glm::reflect(incoming, normal)) < 1e-10f)
    {
        return f0;
    }

    const glm::vec3 wh = glm::normalize(wo + outgoing);
    const float cosThetaO = fabsf(glm::dot(normal, wo));
    const float cosThetaI = fabsf(glm::dot(normal, outgoing));
    const float absCosThetaH = fabsf(glm::dot(normal, wh));
    float woDotWh = fabsf(glm::dot(wo, wh));

    if (cosThetaO <= 0.0f || absCosThetaH <= 0.0f || woDotWh <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    float lambdaO = bsdfTrowbridgeReitzLambda(cosThetaO, roughness);
    float lambdaI = bsdfTrowbridgeReitzLambda(cosThetaI, roughness);
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
        const glm::vec3 crossEdges = glm::cross(triangles.edges1[primitive.index],
            triangles.edges2[primitive.index]);
        const float twiceArea = glm::length(crossEdges);
        if (0.5f * twiceArea <= EPSILON) return sample;
        sample.normal = crossEdges / twiceArea;
        sample.pdfArea = 2.0f / twiceArea;
    }
    else if (primitive.type == SPHERE)
    {
        const Sphere& sphere = primitiveStore.spheres[primitive.index];
        const float z = 1.0f - 2.0f * u01(rng);
        const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = TWO_PI * u01(rng);
        const glm::vec3 localNormal(radial * cosf(phi), radial * sinf(phi), z);
        sample.position = multiplyMV(sphere.transform, glm::vec4(0.5f * localNormal, 1.0f));
        const glm::vec3 transformedNormal = multiplyMV(sphere.invTranspose, glm::vec4(localNormal, 0.0f));
        const float normalLength = glm::length(transformedNormal);
        const float jacobian = fabsf(glm::determinant(glm::mat3(sphere.transform))) * normalLength;
        if (jacobian <= EPSILON) return sample;
        sample.normal = transformedNormal / normalLength;
        sample.pdfArea = 1.0f / (PI * jacobian);
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
        sample.pdfArea = 1.0f / totalArea;
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

template <bool TrianglesOnly>
__device__ __forceinline__ bool isOccluded(const Ray& ray, float maxDistance)
{
    const glm::vec3 inverseDirection = rayInverseDirection(ray);
    const bool fastBounds = hasFiniteRayReciprocals(inverseDirection);
    int nodeIndex = 0;
    while (nodeIndex < traversalData.nodeCount)
    {
        const float4 lower = traversalData.nodes[nodeIndex].lower;
        const float4 upper = traversalData.nodes[nodeIndex].upper;
        const int firstPrimitive = __float_as_int(lower.w);
        const int countOrEscape = __float_as_int(upper.w);
        if (!intersectTraversalNode(ray, inverseDirection, fastBounds, lower, upper, maxDistance))
        {
            nodeIndex = firstPrimitive < 0 ? countOrEscape : nodeIndex + 1;
            continue;
        }
        if (firstPrimitive < 0)
        {
            ++nodeIndex;
            continue;
        }
        for (int offset = 0; offset < countOrEscape; ++offset)
        {
            int unusedPrimitiveIndex;
            glm::vec2 unusedBarycentrics;
            if (intersectLeafPrimitive<TrianglesOnly>(firstPrimitive + offset,
                ray, maxDistance, unusedBarycentrics, unusedPrimitiveIndex) > 0.0f)
                return true;
        }
        ++nodeIndex;
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

__device__ float trowbridgeReitzDistribution(float cosThetaH, float roughness)
{
    const float alpha = glm::clamp(roughness, 0.001f, 1.0f);
    const float alpha2 = alpha * alpha;
    const float denominator = cosThetaH * cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denominator * denominator);
}

__device__ float roughSpecularPdf(const glm::vec3& incoming, const glm::vec3& outgoing,
    glm::vec3 normal, float roughness)
{
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f) normal = -normal;
    const glm::vec3 wo = -glm::normalize(incoming);
    const glm::vec3 wi = glm::normalize(outgoing);
    if (glm::dot(normal, wo) <= 0.0f || glm::dot(normal, wi) <= 0.0f ||
        glm::length2(wo + wi) <= EPSILON) return 0.0f;
    const glm::vec3 wh = glm::normalize(wo + wi);
    const float woDotWh = fabsf(glm::dot(wo, wh));
    const float cosThetaH = fabsf(glm::dot(normal, wh));
    return woDotWh > EPSILON ? trowbridgeReitzDistribution(cosThetaH, roughness) * cosThetaH /
        (4.0f * woDotWh) : 0.0f;
}

__device__ glm::vec3 evaluateDirectBSDF(const Material& material, const glm::vec3& incoming,
    const glm::vec3& outgoing, glm::vec3 normal, float& pdf)
{
    pdf = 0.0f;
    normal = glm::normalize(normal);
    if (glm::dot(incoming, normal) > 0.0f) normal = -normal;
    const glm::vec3 wo = -glm::normalize(incoming);
    const glm::vec3 wi = glm::normalize(outgoing);
    const float cosThetaO = glm::dot(normal, wo);
    const float cosThetaI = glm::dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) return glm::vec3(0.0f);

    if (material.type == MATERIAL_DIFFUSE)
    {
        pdf = cosThetaI / PI;
        return material.color / PI;
    }
    if (material.type != MATERIAL_COOK_TORRANCE && material.type != MATERIAL_MICROFACETS)
        return glm::vec3(0.0f);

    const float roughness = glm::clamp(material.roughness, 0.001f, 1.0f);
    const glm::vec3 wh = glm::normalize(wo + wi);
    const float cosThetaH = fabsf(glm::dot(normal, wh));
    const float woDotWh = fabsf(glm::dot(wo, wh));
    const float D = trowbridgeReitzDistribution(cosThetaH, roughness);
    const float specularPdf = woDotWh > EPSILON && glm::length2(wo + wi) > EPSILON ?
        D * cosThetaH / (4.0f * woDotWh) : 0.0f;
    const float G = 1.0f / (1.0f + bsdfTrowbridgeReitzLambda(cosThetaO, roughness) +
        bsdfTrowbridgeReitzLambda(cosThetaI, roughness));
    // MICROFACETS uses albedo with unit Fresnel.
    const glm::vec3 f0 = material.type == MATERIAL_MICROFACETS ? material.color :
        glm::mix(glm::vec3(0.04f), material.color,
            glm::clamp(material.metallic, 0.0f, 1.0f));
    glm::vec3 result = f0 * (D * G / (4.0f * cosThetaO * cosThetaI));
    pdf = specularPdf;
    if (material.type == MATERIAL_COOK_TORRANCE)
    {
        result += (1.0f - glm::clamp(material.metallic, 0.0f, 1.0f)) * material.color / PI;
        const float specularProbability = glm::clamp(fmaxf(f0.x, fmaxf(f0.y, f0.z)), 0.05f, 0.95f);
        pdf = specularProbability * specularPdf + (1.0f - specularProbability) * cosThetaI / PI;
    }
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
// contribution is applied by traceShadowRays only when the sample is visible.
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
        if (lightPdf <= EPSILON) return;
        float scatteringPdf;
        const glm::vec3 bsdf = evaluateDirectBSDF(material, pathSegment.ray.direction,
            sample.direction, normal, scatteringPdf);
        const float lightPdf2 = lightPdf * lightPdf;
        const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
        shadowRay.ray = { intersectionPoint + 0.001f * offsetNormal, sample.direction };
        shadowRay.maxDistance = FLT_MAX;
        shadowRay.contribution = pathSegment.color * bsdf *
            sample.radiance * (cosSurface * misWeight / lightPdf);
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
    if (lightPdf <= EPSILON) return;
    float scatteringPdf;
    const glm::vec3 bsdf = evaluateDirectBSDF(material, pathSegment.ray.direction, wi, normal, scatteringPdf);
    const float lightPdf2 = lightPdf * lightPdf;
    const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
    const Material& lightMaterial = materials[primitiveMaterialId(light, primitiveStore)];
    shadowRay.ray = { shadowOrigin, wi };
    shadowRay.maxDistance = distance - 0.001f;
    shadowRay.contribution = pathSegment.color * bsdf *
        emittedRadiance(lightMaterial) * (cosSurface * misWeight / lightPdf);
    shadowRay.active = 1;
}

__global__ void prepareShading(
    int iter,
    int depth,
    int num_paths,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    ShadingState* shadingStates,
    const Material* materials,
    DevicePrimitiveStore primitiveStore,
    DeviceEnvironmentMap environment,
    DeviceTextureStore textureStore,
    int lightCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        shadingStates[idx].materialId = -1;
        const ShadeableIntersection& intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            PathSegment& pathSegment = pathSegments[idx];
            const PrimitiveRef& hitPrimitive = primitiveStore.primitives[intersection.primitiveIndex];
            const int materialId = primitiveMaterialId(hitPrimitive, primitiveStore);
            Material material = materials[materialId];
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
                return;
            }

            // hit emitting object
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
				// will be shaded in shadeBSDF kernel
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
                ShadingState& state = shadingStates[idx];
                state.normal = geometricNormal;
                state.color = material.color;
                state.rng = makeSeededRandomEngine(iter, pathSegment.pixelIndex,
                    pathSegment.remainingBounces);
                state.materialId = materialId;
            }
        }
        else {
			// Ray hit nothing, so sample the environment map if present
            PathSegment& pathSegment = pathSegments[idx];
            if (environment.valid())
            {
                const int texelIndex = environmentTexelIndex(environment, pathSegment.ray.direction);
                float misWeight = 1.0f;
                if (!pathSegment.previousBounceWasSpecular && pathSegment.previousBsdfPdf > 0.0f)
                {
                    const float lightPdf = environment.pdfSolidAngle[texelIndex] /
                        static_cast<float>(lightCount);
                    const float bsdfPdf2 = pathSegment.previousBsdfPdf * pathSegment.previousBsdfPdf;
                    misWeight = bsdfPdf2 / (bsdfPdf2 + lightPdf * lightPdf);
                }
                pathSegment.radiance += pathSegment.color *
                    environment.texels[texelIndex] * misWeight;
            }
            pathSegment.remainingBounces = 0;
		}
    }
}

// Sample lights before scattering changes the incoming ray or throughput.
// Carry the advanced engine into shadeBSDF to preserve the sampling sequence,
// including early returns from unsuccessful direct-light samples.
__global__ void generateDirectLighting(
    int numPaths,
    const ShadeableIntersection* intersections,
    const PathSegment* paths,
    ShadingState* shadingStates,
    ShadowRay* shadowRays,
    const Material* materials,
    DevicePrimitiveStore primitiveStore,
    const int* lightPrimitives,
    int emissiveLightCount,
    DeviceEnvironmentMap environment,
    int lightCount)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;
    shadowRays[idx].active = 0;
    ShadingState& state = shadingStates[idx];
    if (state.materialId < 0) return;
    Material material = materials[state.materialId];
    if (material.type == MATERIAL_MIRROR || material.type == MATERIAL_DIELECTRIC) return;
    material.color = state.color;
    const PathSegment& path = paths[idx];
    const glm::vec3 point = path.ray.origin + intersections[idx].t * path.ray.direction;
    const glm::vec3 normal = glm::dot(path.ray.direction, state.normal) < 0.0f ?
        state.normal : -state.normal;
    thrust::default_random_engine rng = state.rng;
    enqueueDirectLighting(material, path, point, normal, primitiveStore, materials,
        lightPrimitives, emissiveLightCount, environment, lightCount, idx, shadowRays[idx], rng);
    state.rng = rng;
}

// Scattering no longer carries texture, light-sampling, or emission temporaries.
__global__ void shadeBSDF(
    int traceDepth,
    int numPaths,
    const ShadeableIntersection* intersections,
    PathSegment* paths,
    const ShadingState* shadingStates,
    const Material* materials)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths || shadingStates[idx].materialId < 0) return;
    const ShadingState& state = shadingStates[idx];
    Material material = materials[state.materialId];
    material.color = state.color;
    PathSegment& pathSegment = paths[idx];
    const glm::vec3 incoming = glm::normalize(pathSegment.ray.direction);
    const glm::vec3 intersect = pathSegment.ray.origin + intersections[idx].t * pathSegment.ray.direction;
    const glm::vec3 geometricNormal = state.normal;
    const bool enteringDielectric = glm::dot(incoming, geometricNormal) < 0.0f;
    const glm::vec3 normal = enteringDielectric ? geometricNormal : -geometricNormal;
    thrust::default_random_engine rng = state.rng;
    switch (material.type)
    {
    case MATERIAL_MIRROR:
        pathSegment.color *= material.color;
        scatterMirror(pathSegment, intersect, normal);
        pathSegment.previousBsdfPdf = 0.0f;
        pathSegment.previousBounceWasSpecular = true;
        break;
    case MATERIAL_DIELECTRIC:
        scatterDielectric(pathSegment, intersect, geometricNormal, material.indexOfRefraction, rng);
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

// Traverse only the visibility rays enqueued by generateDirectLighting.
template <bool TrianglesOnly>
__global__ void traceShadowRays(
    int numShadowRays,
    const ShadowRay* shadowRays,
    const int* rayOrder,
    PathSegment* pathSegments)
{
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= numShadowRays) return;
    const int index = rayOrder ? rayOrder[lane] : lane;

    // rayOrder contains unique active slots, optionally in Morton order. Each
    // path still owns one slot, so accumulation does not need atomics.
    const ShadowRay& shadowRay = shadowRays[index];
    if (!isOccluded<TrianglesOnly>(shadowRay.ray, shadowRay.maxDistance))
    {
        pathSegments[shadowRay.pathIndex].radiance += shadowRay.contribution;
    }
}

#if MATERIAL_SORT
__global__ void buildMaterialSortKeys(
    int num_paths,
    int material_count,
    const ShadeableIntersection* shadeableIntersections,
    const Material* materials,
    DevicePrimitiveStore primitiveStore,
    unsigned int* materialSortKeys,
    int* order)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        order[idx] = idx;
        const ShadeableIntersection& intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            const int materialId = primitiveMaterialId(
                primitiveStore.primitives[intersection.primitiveIndex], primitiveStore);
            materialSortKeys[idx] = static_cast<unsigned int>(materials[materialId].type) *
                static_cast<unsigned int>(material_count) + static_cast<unsigned int>(materialId);
        }
        else
        {
            materialSortKeys[idx] = static_cast<unsigned int>(MATERIAL_TYPE_COUNT) *
                static_cast<unsigned int>(material_count);
        }
    }
}

// Radix passes move only key/index pairs. Gather the large payloads once, using
// the same permutation for paths and intersections so they cannot get unpaired.
__global__ void gatherMaterialSortedPaths(int count, const int* order,
    const PathSegment* paths, const ShadeableIntersection* intersections,
    PathSegment* sortedPaths, ShadeableIntersection* sortedIntersections)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int source = order[index];
    sortedPaths[index] = paths[source];
    sortedIntersections[index] = intersections[source];
}
#endif

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
    glm::vec3 boundsMin, glm::vec3 inverseExtent, const int* inputOrder,
    unsigned int* keys, int* order)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    // inputOrder may alias order: each thread only reads/writes its own slot.
    const int source = inputOrder ? inputOrder[index] : index;
    const Ray& ray = rays[source].ray;
    const unsigned int origin = mortonCode((ray.origin - boundsMin) * inverseExtent, 64.0f);
    const unsigned int direction = mortonCode(0.5f * ray.direction + glm::vec3(0.5f), 16.0f);
    keys[index] = (origin << 12) | direction;
    order[index] = source;
}

template <typename RayPayload>
const int* orderRays(int count, const RayPayload* rays, int blockSize, const int* inputOrder = NULL)
{
    // NULL means identity for closest-hit rays. Shadows must retain their
    // compacted active-index list even when Morton sorting is skipped.
    if (count < MORTON_SORT_MIN_RAYS) return inputOrder;
    buildMortonSortKeys<<<(count + blockSize - 1) / blockSize, blockSize>>>(
        count, rays, mortonBoundsMin, mortonInverseExtent, inputOrder, dev_sortKeys[0], dev_sortIndices[0]);
    const int* order = sortRayIndices(count, mortonSortBits);
    checkCUDAError("sort traversal rays by Morton code");
    return order;
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
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = num_paths == 0;
    while (!iterationComplete)
    {
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        const int* traversalOrder = NULL;
#if MORTON_SORT
        traversalOrder = orderRays(num_paths, dev_paths, blockSize1d);
#endif
        if (trianglesOnly)
            computeIntersections<true><<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_paths, traversalOrder, dev_intersections);
        else
            computeIntersections<false><<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_paths, traversalOrder, dev_intersections);
        checkCUDAError("trace one bounce");
        depth++;

        // Sort before preparing surfaces: the shading scratch buffer and shadow
        // ownership use these indices until the end of the bounce.

#if MATERIAL_SORT
        buildMaterialSortKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            hst_scene->materials.size(),
            dev_intersections,
            dev_materials,
            dev_primitiveStore,
            dev_sortKeys[0],
            dev_sortIndices[0]
        );
        checkCUDAError("build material sort keys");

        const int* materialOrder = sortRayIndices(num_paths, materialSortBits);
        gatherMaterialSortedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, materialOrder, dev_paths, dev_intersections,
            dev_pathsAlternate, dev_intersectionsAlternate);
        std::swap(dev_paths, dev_pathsAlternate);
        std::swap(dev_intersections, dev_intersectionsAlternate);
        checkCUDAError("sort paths by material type");

#endif
        prepareShading<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_shadingStates,
            dev_materials,
            dev_primitiveStore,
            dev_environment,
            dev_textureStore,
            totalLightCount
        );
        checkCUDAError("prepare shading surfaces");

        if (totalLightCount > 0)
        {
            generateDirectLighting<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_intersections, dev_paths, dev_shadingStates,
                dev_shadowRays, dev_materials, dev_primitiveStore, dev_lightPrimitives,
                dev_lightPrimitiveCount, dev_environment, totalLightCount);
            checkCUDAError("enqueue direct lighting");
        }

        shadeBSDF<<<numblocksPathSegmentTracing, blockSize1d>>>(
            traceDepth, num_paths, dev_intersections, dev_paths, dev_shadingStates, dev_materials);
        checkCUDAError("scatter paths");

        // Compact just active indices; shadow payloads and their path owners
        // remain in place until visibility contributions have been accumulated.
        int numShadowRays = 0;
        if (totalLightCount > 0)
        {
            numShadowRays = compactShadowIndices(num_paths);
            checkCUDAError("compact shadow indices");
        }

        if (numShadowRays > 0)
        {
            const dim3 shadowBlocks = (numShadowRays + blockSize1d - 1) / blockSize1d;
            const int* shadowOrder = dev_sortIndices[0];
#if MORTON_SORT
            shadowOrder = orderRays(numShadowRays, dev_shadowRays, blockSize1d, shadowOrder);
#endif
            if (trianglesOnly)
                traceShadowRays<true><<<shadowBlocks, blockSize1d>>>(
                    numShadowRays, dev_shadowRays, shadowOrder, dev_paths);
            else
                traceShadowRays<false><<<shadowBlocks, blockSize1d>>>(
                    numShadowRays, dev_shadowRays, shadowOrder, dev_paths);
            checkCUDAError("trace and accumulate direct lighting");
        }

        gatherTerminatedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_image,
            dev_paths
        );
        checkCUDAError("gather terminated paths");

        // Select live paths out of place, then swap pointers instead of copying
        // the compacted records back. Shadows must finish before this swap.
        num_paths = compactPaths(num_paths);
        checkCUDAError("compact active paths");

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

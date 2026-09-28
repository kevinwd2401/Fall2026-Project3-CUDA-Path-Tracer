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

#define MATERIAL_SORT 0
#define RUSSIAN_ROULETTE 1
#define RUSSIAN_ROULETTE_START_DEPTH 3
#define DEPTH_OF_FIELD 1
#define ERRORCHECK 1

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
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = NULL;
static int* dev_bvhPrimitiveIndices = NULL;
static Material* dev_materials = NULL;
static int* dev_lightPrimitives = NULL;
static int dev_lightPrimitiveCount = 0;
static glm::vec3* dev_environmentTexels = NULL;
static float* dev_environmentCdf = NULL;
static float* dev_environmentPdfSolidAngle = NULL;
static TextureInfo* dev_textures = NULL;
static glm::vec4* dev_textureTexels = NULL;
static int dev_textureCount = 0;

struct DeviceEnvironmentMap
{
    const glm::vec3* texels = NULL;
    const float* cdf = NULL;
    const float* pdfSolidAngle = NULL;
    int width = 0;
    int height = 0;

    __host__ __device__ bool valid() const
    {
        return texels != NULL && cdf != NULL && pdfSolidAngle != NULL && width > 0 && height > 0;
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
    const Triangle* triangles = NULL;
};

static DevicePrimitiveStore dev_primitiveStore;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialSortKeys = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }
    dev_primitiveStore = { dev_primitives, dev_cubes, dev_spheres, dev_triangles };

    if (!scene->bvhNodes.empty())
    {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_bvhPrimitiveIndices, scene->bvhPrimitiveIndices.size() * sizeof(int));
        cudaMemcpy(dev_bvhPrimitiveIndices, scene->bvhPrimitiveIndices.data(),
            scene->bvhPrimitiveIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }

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
        cudaMalloc(&dev_environmentCdf, environmentPixelCount * sizeof(float));
        cudaMalloc(&dev_environmentPdfSolidAngle, environmentPixelCount * sizeof(float));
        cudaMemcpy(dev_environmentTexels, scene->environment.texels.data(),
            environmentPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_environmentCdf, scene->environment.cdf.data(),
            environmentPixelCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_environmentPdfSolidAngle, scene->environment.pdfSolidAngle.data(),
            environmentPixelCount * sizeof(float), cudaMemcpyHostToDevice);
        dev_environment = { dev_environmentTexels, dev_environmentCdf, dev_environmentPdfSolidAngle,
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

    cudaMalloc(&dev_materialSortKeys, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_primitives);
    cudaFree(dev_cubes);
    cudaFree(dev_spheres);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhPrimitiveIndices);
    cudaFree(dev_materials);
    cudaFree(dev_lightPrimitives);
    cudaFree(dev_environmentTexels);
    cudaFree(dev_environmentCdf);
    cudaFree(dev_environmentPdfSolidAngle);
    cudaFree(dev_textures);
    cudaFree(dev_textureTexels);
    cudaFree(dev_intersections);
    cudaFree(dev_materialSortKeys);
    dev_environmentTexels = NULL;
    dev_environmentCdf = NULL;
    dev_environmentPdfSolidAngle = NULL;
    dev_environment = DeviceEnvironmentMap{};
    dev_textures = NULL;
    dev_textureTexels = NULL;
    dev_textureCount = 0;
    dev_textureStore = DeviceTextureStore{};
    dev_primitives = NULL;
    dev_cubes = NULL;
    dev_spheres = NULL;
    dev_triangles = NULL;
    dev_primitiveStore = DevicePrimitiveStore{};

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

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__device__ float intersectPrimitive(const PrimitiveRef& primitive, const DevicePrimitiveStore& primitiveStore,
    const Ray& ray, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside)
{
    switch (primitive.type)
    {
    case CUBE:
        return boxIntersectionTest(primitiveStore.cubes[primitive.index], ray, intersectionPoint, normal, outside);
    case SPHERE:
        return sphereIntersectionTest(primitiveStore.spheres[primitive.index], ray, intersectionPoint, normal, outside);
    case TRIANGLE:
        return triangleIntersectionTest(primitiveStore.triangles[primitive.index], ray, intersectionPoint, normal, uv, outside);
    }
    return -1.0f;
}

__device__ int primitiveMaterialId(const PrimitiveRef& primitive, const DevicePrimitiveStore& primitiveStore)
{
    switch (primitive.type)
    {
    case CUBE: return primitiveStore.cubes[primitive.index].materialid;
    case SPHERE: return primitiveStore.spheres[primitive.index].materialid;
    case TRIANGLE: return primitiveStore.triangles[primitive.index].materialid;
    }
    return -1;
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    DevicePrimitiveStore primitiveStore,
    const BVHNode* bvhNodes,
    int bvhNodeCount,
    const int* bvhPrimitiveIndices,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_primitive_index = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 hit_uv(0.0f);


        //bvh traversal
        int nodeIndex = 0;
        while (nodeIndex < bvhNodeCount)
        {
            const BVHNode& node = bvhNodes[nodeIndex];
            float boxEntryDistance;
            if (!aabbIntersectionTest(pathSegment.ray, node.boundsMin, node.boundsMax,
                t_min, boxEntryDistance))
            {
                nodeIndex = node.escapeIndex;
                continue;
            }

            if (node.primitiveCount == 0)
            {
                // Nodes are stored depth-first, so the left child is next.
                ++nodeIndex;
                continue;
            }

            for (int primitiveOffset = 0; primitiveOffset < node.primitiveCount; ++primitiveOffset)
            {
                const int primitiveIndex = bvhPrimitiveIndices[node.firstPrimitive + primitiveOffset];
                const PrimitiveRef& primitive = primitiveStore.primitives[primitiveIndex];
                bool outside = true;
                float t = -1.0f;
                glm::vec2 candidateUV(0.0f);

                t = intersectPrimitive(primitive, primitiveStore, pathSegment.ray,
                    tmp_intersect, tmp_normal, candidateUV, outside);

                if (t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_primitive_index = primitiveIndex;
                    normal = tmp_normal;
                    hit_uv = candidateUV;
                }
            }
            nodeIndex = node.escapeIndex;
        }

        if (hit_primitive_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].primitiveIndex = -1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = primitiveMaterialId(
                primitiveStore.primitives[hit_primitive_index], primitiveStore);
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceUV = hit_uv;
            intersections[path_index].primitiveIndex = hit_primitive_index;
        }
    }
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
        const Triangle& triangle = primitiveStore.triangles[primitive.index];
        return glm::normalize(glm::cross(
            triangle.triangleVertices[1] - triangle.triangleVertices[0],
            triangle.triangleVertices[2] - triangle.triangleVertices[0]));
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
        const Triangle& triangle = primitiveStore.triangles[primitive.index];
        const float area = 0.5f * glm::length(glm::cross(
            triangle.triangleVertices[1] - triangle.triangleVertices[0],
            triangle.triangleVertices[2] - triangle.triangleVertices[0]));
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
        const Triangle& triangle = primitiveStore.triangles[primitive.index];
        const float rootU = sqrtf(u01(rng));
        const float v = u01(rng);
        sample.position = (1.0f - rootU) * triangle.triangleVertices[0] +
            rootU * (1.0f - v) * triangle.triangleVertices[1] + rootU * v * triangle.triangleVertices[2];
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
    const float chooseTexel = fminf(u01(rng), 0.99999994f);
    int low = 0;
    int high = environment.width * environment.height - 1;
    while (low < high)
    {
        const int middle = low + (high - low) / 2;
        if (environment.cdf[middle] >= chooseTexel) high = middle;
        else low = middle + 1;
    }

    const int texelIndex = low;
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

__device__ bool isOccluded(const Ray& ray, float maxDistance, const DevicePrimitiveStore& primitiveStore,
    const BVHNode* bvhNodes, int bvhNodeCount, const int* bvhPrimitiveIndices)
{
    int nodeIndex = 0;
    while (nodeIndex < bvhNodeCount)
    {
        const BVHNode& node = bvhNodes[nodeIndex];
        float entryDistance;
        if (!aabbIntersectionTest(ray, node.boundsMin, node.boundsMax, maxDistance, entryDistance))
        {
            nodeIndex = node.escapeIndex;
            continue;
        }
        if (node.primitiveCount == 0)
        {
            ++nodeIndex;
            continue;
        }
        for (int offset = 0; offset < node.primitiveCount; ++offset)
        {
            const PrimitiveRef& primitive = primitiveStore.primitives[
                bvhPrimitiveIndices[node.firstPrimitive + offset]];
            glm::vec3 ignoredPoint, ignoredNormal;
            glm::vec2 ignoredUV;
            bool outside;
            const float t = intersectPrimitive(primitive, primitiveStore, ray,
                ignoredPoint, ignoredNormal, ignoredUV, outside);
            if (t > 1e-4f && t < maxDistance) return true;
        }
        nodeIndex = node.escapeIndex;
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

__device__ glm::vec3 sampleBaseColorTexture(const Material& material, const glm::vec2& uv,
    const DeviceTextureStore& textureStore)
{
    if (!textureStore.validTextureIndex(material.baseColorTexture)) return glm::vec3(1.0f);
    const TextureInfo& texture = textureStore.textures[material.baseColorTexture];
    const glm::vec3 srgb = glm::vec3(sampleTexture(texture, uv, textureStore));
    return glm::vec3(srgbToLinear(srgb.x), srgbToLinear(srgb.y), srgbToLinear(srgb.z));
}

__device__ glm::vec3 sampleNormalTexture(const Material& material, const Triangle& triangle,
    const glm::vec2& uv, glm::vec3 geometricNormal, const DeviceTextureStore& textureStore)
{
    if (!textureStore.validTextureIndex(material.normalTexture) || !triangle.hasTextureCoordinates)
        return geometricNormal;

    const TextureInfo& texture = textureStore.textures[material.normalTexture];
    const glm::vec3 encodedNormal = glm::vec3(sampleTexture(texture, uv, textureStore));
    glm::vec3 tangentSpaceNormal = 2.0f * encodedNormal - glm::vec3(1.0f);
    tangentSpaceNormal.x *= material.normalScale;
    tangentSpaceNormal.y *= material.normalScale;
    if (glm::length2(tangentSpaceNormal) <= EPSILON) return geometricNormal;
    tangentSpaceNormal = glm::normalize(tangentSpaceNormal);

    const glm::vec3 edge1 = triangle.triangleVertices[1] - triangle.triangleVertices[0];
    const glm::vec3 edge2 = triangle.triangleVertices[2] - triangle.triangleVertices[0];
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

__device__ glm::vec3 estimateDirectLighting(const Material& material, const PathSegment& pathSegment,
    const glm::vec3& intersectionPoint, const glm::vec3& normal, const DevicePrimitiveStore& primitiveStore,
    const Material* materials, const int* lightPrimitives, int emissiveLightCount,
    const DeviceEnvironmentMap& environment, int totalLightCount, const BVHNode* bvhNodes,
    int bvhNodeCount, const int* bvhPrimitiveIndices, thrust::default_random_engine& rng)
{
    if (totalLightCount <= 0) return glm::vec3(0.0f);
    thrust::uniform_int_distribution<int> chooseLight(0, totalLightCount - 1);
    const int chosenLight = chooseLight(rng);
    const glm::vec3 offsetNormal = glm::dot(pathSegment.ray.direction, normal) < 0.0f ? normal : -normal;

    if (chosenLight == emissiveLightCount)
    {
        const EnvironmentSample sample = sampleEnvironment(environment, rng);
        if (!sample.valid) return glm::vec3(0.0f);
        const float cosSurface = fmaxf(0.0f, glm::dot(offsetNormal, sample.direction));
        if (cosSurface <= 0.0f) return glm::vec3(0.0f);
        const Ray shadowRay{ intersectionPoint + 0.001f * offsetNormal, sample.direction };
        if (isOccluded(shadowRay, FLT_MAX, primitiveStore, bvhNodes, bvhNodeCount, bvhPrimitiveIndices))
            return glm::vec3(0.0f);
        const float lightPdf = sample.pdfSolidAngle / static_cast<float>(totalLightCount);
        const float scatteringPdf = bsdfPdf(material, pathSegment.ray.direction, sample.direction, normal);
        if (lightPdf <= EPSILON) return glm::vec3(0.0f);
        const float lightPdf2 = lightPdf * lightPdf;
        const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
        return evaluateDirectBSDF(material, pathSegment.ray.direction, sample.direction, normal) *
            sample.radiance * (cosSurface * misWeight / lightPdf);
    }

    // The only non-geometry entry is the environment entry immediately after
    // the emissive primitives, so a valid selection below is always a surface.
    if (chosenLight >= emissiveLightCount) return glm::vec3(0.0f);
    const PrimitiveRef& light = primitiveStore.primitives[lightPrimitives[chosenLight]];
    const LightSurfaceSample sample = sampleLightSurface(light, primitiveStore, rng);
    if (!sample.valid) return glm::vec3(0.0f);

    const glm::vec3 shadowOrigin = intersectionPoint + 0.001f * offsetNormal;
    const glm::vec3 toLight = sample.position - shadowOrigin;
    const float distance2 = glm::dot(toLight, toLight);
    if (distance2 <= EPSILON) return glm::vec3(0.0f);
    const float distance = sqrtf(distance2);
    const glm::vec3 wi = toLight / distance;
    const float cosSurface = fmaxf(0.0f, glm::dot(offsetNormal, wi));
    // Emissive primitives radiate from their front side only.  Their front
    // side is the outward primitive normal (or triangle winding normal).
    const float cosLight = glm::dot(sample.normal, -wi);
    if (cosSurface <= 0.0f || cosLight <= EPSILON) return glm::vec3(0.0f);

    Ray shadowRay{ shadowOrigin, wi };
    if (isOccluded(shadowRay, distance - 0.001f, primitiveStore, bvhNodes, bvhNodeCount, bvhPrimitiveIndices))
        return glm::vec3(0.0f);

    const float lightPdf = distance2 * sample.pdfArea /
        (cosLight * static_cast<float>(totalLightCount));
    const float scatteringPdf = bsdfPdf(material, pathSegment.ray.direction, wi, normal);
    if (lightPdf <= EPSILON) return glm::vec3(0.0f);
    const float lightPdf2 = lightPdf * lightPdf;
    const float misWeight = lightPdf2 / (lightPdf2 + scatteringPdf * scatteringPdf);
    const Material& lightMaterial = materials[primitiveMaterialId(light, primitiveStore)];
    return evaluateDirectBSDF(material, pathSegment.ray.direction, wi, normal) *
        emittedRadiance(lightMaterial) * (cosSurface * misWeight / lightPdf);
}

__global__ void shadeBSDF(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    DevicePrimitiveStore primitiveStore,
    const int* lightPrimitives,
    int emissiveLightCount,
    DeviceEnvironmentMap environment,
    DeviceTextureStore textureStore,
    int lightCount,
    const BVHNode* bvhNodes,
    int bvhNodeCount,
    const int* bvhPrimitiveIndices)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            PathSegment pathSegment = pathSegments[idx];
            const PrimitiveRef& hitPrimitive = primitiveStore.primitives[intersection.primitiveIndex];
            if (hitPrimitive.type == TRIANGLE &&
                primitiveStore.triangles[hitPrimitive.index].hasTextureCoordinates)
            {
                material.color *= sampleBaseColorTexture(material, intersection.surfaceUV, textureStore);
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
                glm::vec3 geometricNormal = glm::normalize(intersection.surfaceNormal);
                if (hitPrimitive.type == TRIANGLE &&
                    primitiveStore.triangles[hitPrimitive.index].hasTextureCoordinates)
                {
                    geometricNormal = sampleNormalTexture(material, primitiveStore.triangles[hitPrimitive.index], intersection.surfaceUV,
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
                // One uniformly selected emissive primitive is sampled at
                // every non-delta vertex.  The light and BSDF PDFs share the
                // power heuristic, with the matching BSDF-hit weight above.
                if (material.type != MATERIAL_MIRROR && material.type != MATERIAL_DIELECTRIC)
                {
                    pathSegment.radiance += pathSegment.color * estimateDirectLighting(material,
                        pathSegment, intersect, normal, primitiveStore, materials, lightPrimitives, emissiveLightCount,
                        environment, lightCount,
                        bvhNodes, bvhNodeCount, bvhPrimitiveIndices, rng);
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
                if (depth >= RUSSIAN_ROULETTE_START_DEPTH && pathSegment.remainingBounces > 0)
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

__global__ void buildMaterialSortKeys(
    int num_paths,
    int material_count,
    ShadeableIntersection* shadeableIntersections,
    Material* materials,
    int* materialSortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            materialSortKeys[idx] = static_cast<int>(material.type) * material_count + intersection.materialId;
        }
        else
        {
            materialSortKeys[idx] = MATERIAL_TYPE_COUNT * material_count;
        }
    }
}

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

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    const float focalLength = guiData ? guiData->FocalLength : 2.0f;
    const float lensRadius = guiData ? guiData->LensRadius : 0.008f;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, focalLength, lensRadius);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_primitiveStore,
            dev_bvhNodes,
            static_cast<int>(hst_scene->bvhNodes.size()),
            dev_bvhPrimitiveIndices,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
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
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_primitiveStore,
            dev_lightPrimitives,
            dev_lightPrimitiveCount,
            dev_environment,
            dev_textureStore,
            totalLightCount,
            dev_bvhNodes,
            static_cast<int>(hst_scene->bvhNodes.size()),
            dev_bvhPrimitiveIndices
        );
        checkCUDAError("shade materials");

        gatherTerminatedPaths<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_image,
            dev_paths
        );
        checkCUDAError("gather terminated paths");

        dev_path_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            IsTerminated());
        num_paths = dev_path_end - dev_paths;

		iterationComplete = (depth >= traceDepth) || (num_paths == 0);

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

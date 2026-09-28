#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "json.hpp"
#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace
{
constexpr uint32_t GLB_MAGIC = 0x46546C67;
constexpr uint32_t GLB_JSON_CHUNK = 0x4E4F534A;
constexpr uint32_t GLB_BIN_CHUNK = 0x004E4942;
constexpr int GLTF_MODE_TRIANGLES = 4;
constexpr int BVH_LEAF_SIZE = 4;
constexpr int BVH_SAH_BINS = 16;

struct Bounds
{
    glm::vec3 minimum = glm::vec3(FLT_MAX);
    glm::vec3 maximum = glm::vec3(-FLT_MAX);

    void grow(const glm::vec3& point)
    {
        minimum = glm::min(minimum, point);
        maximum = glm::max(maximum, point);
    }

    void grow(const Bounds& other)
    {
        minimum = glm::min(minimum, other.minimum);
        maximum = glm::max(maximum, other.maximum);
    }

    float surfaceArea() const
    {
        const glm::vec3 extent = glm::max(maximum - minimum, glm::vec3(0.0f));
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
};

struct BVHBin
{
    Bounds bounds;
    int count = 0;
};

Bounds boundsForGeom(const Geom& geom)
{
    Bounds bounds;
    if (geom.type == TRIANGLE)
    {
        bounds.grow(geom.triangleVertices[0]);
        bounds.grow(geom.triangleVertices[1]);
        bounds.grow(geom.triangleVertices[2]);
        return bounds;
    }

    for (int corner = 0; corner < 8; ++corner)
    {
        const glm::vec3 local(
            (corner & 1) ? 0.5f : -0.5f,
            (corner & 2) ? 0.5f : -0.5f,
            (corner & 4) ? 0.5f : -0.5f);
        bounds.grow(glm::vec3(geom.transform * glm::vec4(local, 1.0f)));
    }
    return bounds;
}

MaterialType parseMaterialType(const std::string& type)
{
    if (type == "Cook-Torrance" || type == "CookTorrance" || type == "cook-torrance" || type == "Plastic" || type == "Metallic") return MATERIAL_COOK_TORRANCE;
    if (type == "Dielectric" || type == "dielectric" || type == "Refractive" || type == "refractive" || type == "Glass" || type == "glass") return MATERIAL_DIELECTRIC;
    if (type == "Mirror" || type == "mirror" || type == "Specular") return MATERIAL_MIRROR;
    if (type == "Microfacets" || type == "microfacets" || type == "Microfacet" || type == "microfacet") return MATERIAL_MICROFACETS;
    if (type == "Emissive" || type == "emissive" || type == "Emitting" || type == "emitting") return MATERIAL_EMISSIVE;
    return MATERIAL_DIFFUSE;
}

Material makeDefaultMaterial()
{
    Material material{};
    material.color = glm::vec3(1.0f);
    material.emission = glm::vec3(0.0f);
    material.type = MATERIAL_DIFFUSE;
    material.indexOfRefraction = 1.55f;
    material.roughness = 1.0f;
    material.baseColorTexture = -1;
    material.normalTexture = -1;
    material.normalScale = 1.0f;
    return material;
}

float maxComponent(const glm::vec3& v)
{
    return std::max(v.x, std::max(v.y, v.z));
}

bool materialEmits(const Material& material)
{
    return material.type == MATERIAL_EMISSIVE || material.emittance > 0.0f || maxComponent(material.emission) > 0.0f;
}

void finalizeCamera(Camera& camera, RenderState& state, float yscaled)
{
    camera.view = glm::normalize(camera.lookAt - camera.position);
    if (glm::length(camera.view) < EPSILON) camera.view = glm::vec3(0.0f, 0.0f, -1.0f);
    camera.right = glm::cross(camera.view, camera.up);
    if (glm::length(camera.right) < EPSILON) camera.right = glm::vec3(1.0f, 0.0f, 0.0f);
    else camera.right = glm::normalize(camera.right);
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    const float xscaled = yscaled * static_cast<float>(camera.resolution.x) / static_cast<float>(camera.resolution.y);
    camera.fov = glm::vec2(atan(xscaled) * 180.0f / PI, atan(yscaled) * 180.0f / PI);
    camera.pixelLength = glm::vec2(2.0f * xscaled / camera.resolution.x, 2.0f * yscaled / camera.resolution.y);
    state.image.assign(camera.resolution.x * camera.resolution.y, glm::vec3(0.0f));
}

string lowercase(string value)
{
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return value;
}

vector<unsigned char> readBinaryFile(const filesystem::path& path)
{
    ifstream input(path, ios::binary);
    if (!input) throw runtime_error("could not open " + path.string());
    return vector<unsigned char>(istreambuf_iterator<char>(input), istreambuf_iterator<char>());
}

uint32_t readLE32(const vector<unsigned char>& bytes, size_t offset)
{
    if (offset + 4 > bytes.size()) throw runtime_error("truncated GLB");
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
        (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

vector<unsigned char> decodeBase64(const string& encoded)
{
    static const string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    vector<unsigned char> result;
    int value = 0, bits = -8;
    for (unsigned char c : encoded)
    {
        if (isspace(c)) continue;
        if (c == '=') break;
        const size_t digit = alphabet.find(static_cast<char>(c));
        if (digit == string::npos) throw runtime_error("invalid base64 URI");
        value = (value << 6) + static_cast<int>(digit);
        bits += 6;
        if (bits >= 0) { result.push_back(static_cast<unsigned char>((value >> bits) & 0xff)); bits -= 8; }
    }
    return result;
}

size_t componentSize(int type)
{
    switch (type) { case 5120: case 5121: return 1; case 5122: case 5123: return 2; case 5125: case 5126: return 4; default: throw runtime_error("unsupported accessor component type"); }
}

int componentCount(const string& type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    throw runtime_error("unsupported accessor type");
}

struct AccessorView
{
    const vector<unsigned char>* buffer;
    size_t offset, stride, count;
    int componentType, components;
};

AccessorView getAccessor(const json& document, const vector<vector<unsigned char>>& buffers, int accessorIndex)
{
    const auto& accessors = document.at("accessors");
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(accessors.size())) throw runtime_error("accessor index out of range");
    const json& accessor = accessors.at(accessorIndex);
    if (accessor.contains("sparse") || !accessor.contains("bufferView")) throw runtime_error("sparse or bufferless accessors are unsupported");
    const auto& views = document.at("bufferViews");
    const int viewIndex = accessor.at("bufferView").get<int>();
    if (viewIndex < 0 || viewIndex >= static_cast<int>(views.size())) throw runtime_error("buffer view index out of range");
    const json& view = views.at(viewIndex);
    const int bufferIndex = view.at("buffer").get<int>();
    if (bufferIndex < 0 || bufferIndex >= static_cast<int>(buffers.size())) throw runtime_error("buffer index out of range");
    AccessorView result{};
    result.buffer = &buffers.at(bufferIndex);
    result.componentType = accessor.at("componentType").get<int>();
    result.components = componentCount(accessor.at("type").get<string>());
    result.count = accessor.at("count").get<size_t>();
    const size_t packedSize = componentSize(result.componentType) * result.components;
    result.stride = view.value("byteStride", packedSize);
    result.offset = view.value("byteOffset", size_t(0)) + accessor.value("byteOffset", size_t(0));
    if (result.stride < packedSize || (result.count &&
        (result.offset > result.buffer->size() || result.stride > (result.buffer->size() - result.offset) / (result.count - 1) ||
         packedSize > result.buffer->size() - result.offset - result.stride * (result.count - 1)))) throw runtime_error("accessor exceeds its buffer");
    return result;
}

vector<glm::vec3> readVec3(const json& document, const vector<vector<unsigned char>>& buffers, int accessorIndex)
{
    const AccessorView view = getAccessor(document, buffers, accessorIndex);
    if (view.componentType != 5126 || view.components != 3) throw runtime_error("POSITION/NORMAL must be FLOAT VEC3");
    vector<glm::vec3> values(view.count);
    for (size_t i = 0; i < view.count; ++i)
    {
        float f[3];
        memcpy(f, view.buffer->data() + view.offset + i * view.stride, sizeof(f));
        values[i] = glm::vec3(f[0], f[1], f[2]);
    }
    return values;
}

vector<glm::vec2> readVec2(const json& document, const vector<vector<unsigned char>>& buffers, int accessorIndex)
{
    const AccessorView view = getAccessor(document, buffers, accessorIndex);
    if (view.componentType != 5126 || view.components != 2) throw runtime_error("TEXCOORD_0 must be FLOAT VEC2");
    vector<glm::vec2> values(view.count);
    for (size_t i = 0; i < view.count; ++i)
    {
        float f[2];
        memcpy(f, view.buffer->data() + view.offset + i * view.stride, sizeof(f));
        values[i] = glm::vec2(f[0], f[1]);
    }
    return values;
}

vector<uint32_t> readIndices(const json& document, const vector<vector<unsigned char>>& buffers, int accessorIndex)
{
    const AccessorView view = getAccessor(document, buffers, accessorIndex);
    if (view.components != 1) throw runtime_error("indices must be SCALAR");
    vector<uint32_t> values(view.count);
    for (size_t i = 0; i < view.count; ++i)
    {
        const unsigned char* source = view.buffer->data() + view.offset + i * view.stride;
        if (view.componentType == 5121) values[i] = *source;
        else if (view.componentType == 5123) { uint16_t value; memcpy(&value, source, sizeof(value)); values[i] = value; }
        else if (view.componentType == 5125) { uint32_t value; memcpy(&value, source, sizeof(value)); values[i] = value; }
        else throw runtime_error("indices must use an unsigned integer component type");
    }
    return values;
}

glm::mat4 getNodeTransform(const json& node)
{
    if (node.contains("matrix"))
    {
        const auto& values = node.at("matrix");
        if (values.size() != 16) throw runtime_error("node matrix must have 16 values");
        glm::mat4 matrix(1.0f);
        for (int column = 0; column < 4; ++column) for (int row = 0; row < 4; ++row) matrix[column][row] = values.at(column * 4 + row).get<float>();
        return matrix;
    }
    glm::vec3 translation(0.0f), scale(1.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.contains("translation")) { const auto& v = node.at("translation"); translation = glm::vec3(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>()); }
    if (node.contains("scale")) { const auto& v = node.at("scale"); scale = glm::vec3(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>()); }
    if (node.contains("rotation")) { const auto& v = node.at("rotation"); rotation = glm::quat(v.at(3).get<float>(), v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>()); }
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}
}

Scene::Scene(string filename, string environmentFilename)
{
    cout << "Reading scene from " << filename << " ..." << endl << " " << endl;
    const string extension = lowercase(filesystem::path(filename).extension().string());
    if (extension == ".json") loadFromJSON(filename);
    else if (extension == ".gltf" || extension == ".glb") loadFromGLTF(filename);
    else { cout << "Couldn't read from " << filename << endl; exit(-1); }
    if (!environmentFilename.empty()) loadEnvironmentMap(environmentFilename);
    buildBVH();
}

void Scene::loadEnvironmentMap(const string& filename)
{
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    float* pixels = stbi_loadf(filename.c_str(), &width, &height, &sourceChannels, 3);
    if (pixels == nullptr || width <= 0 || height <= 0)
    {
        const string reason = stbi_failure_reason() ? stbi_failure_reason() : "unknown image loading error";
        if (pixels) stbi_image_free(pixels);
        throw runtime_error("could not load HDRI environment map '" + filename + "': " + reason);
    }

    environment.width = width;
    environment.height = height;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    environment.texels.resize(pixelCount);
    environment.cdf.resize(pixelCount);
    environment.pdfSolidAngle.resize(pixelCount);

    // Each texel represents a different solid angle in latitude-longitude
    // space.  Weighting luminance by that angle makes the distribution a PDF
    // over directions rather than a PDF over image pixels.
    vector<float> weights(pixelCount);
    double totalWeight = 0.0;
    const float deltaPhi = TWO_PI / static_cast<float>(width);
    for (int y = 0; y < height; ++y)
    {
        const float theta0 = PI * static_cast<float>(y) / static_cast<float>(height);
        const float theta1 = PI * static_cast<float>(y + 1) / static_cast<float>(height);
        const float texelSolidAngle = deltaPhi * (cosf(theta0) - cosf(theta1));
        for (int x = 0; x < width; ++x)
        {
            const size_t index = static_cast<size_t>(y) * width + x;
            const float* pixel = pixels + index * 3;
            const glm::vec3 color(fmaxf(0.0f, pixel[0]), fmaxf(0.0f, pixel[1]), fmaxf(0.0f, pixel[2]));
            environment.texels[index] = color;
            const float luminance = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
            weights[index] = luminance * texelSolidAngle;
            totalWeight += weights[index];
        }
    }
    stbi_image_free(pixels);

    // A black map has no meaningful luminance distribution.  Fall back to a
    // uniform-sphere distribution so its PDF is still valid and finite.
    if (totalWeight <= 0.0)
    {
        totalWeight = 0.0;
        for (int y = 0; y < height; ++y)
        {
            const float theta0 = PI * static_cast<float>(y) / static_cast<float>(height);
            const float theta1 = PI * static_cast<float>(y + 1) / static_cast<float>(height);
            const float texelSolidAngle = deltaPhi * (cosf(theta0) - cosf(theta1));
            for (int x = 0; x < width; ++x)
            {
                const size_t index = static_cast<size_t>(y) * width + x;
                weights[index] = texelSolidAngle;
                totalWeight += weights[index];
            }
        }
    }

    double cumulativeWeight = 0.0;
    for (size_t index = 0; index < pixelCount; ++index)
    {
        cumulativeWeight += weights[index];
        environment.cdf[index] = static_cast<float>(cumulativeWeight / totalWeight);
        const int y = static_cast<int>(index / width);
        const float theta0 = PI * static_cast<float>(y) / static_cast<float>(height);
        const float theta1 = PI * static_cast<float>(y + 1) / static_cast<float>(height);
        const float texelSolidAngle = deltaPhi * (cosf(theta0) - cosf(theta1));
        environment.pdfSolidAngle[index] = texelSolidAngle > 0.0f ?
            weights[index] / static_cast<float>(totalWeight) / texelSolidAngle : 0.0f;
    }
    environment.cdf.back() = 1.0f;
    cout << "Loaded HDRI environment map " << filename << " (" << width << "x" << height
         << ", importance sampled)." << endl;
}

void Scene::rebuildEmissivePrimitives()
{
    emissivePrimitives.clear();
    for (size_t i = 0; i < geoms.size(); ++i)
    {
        const int materialID = geoms[i].materialid;
        if (materialID >= 0 && materialID < static_cast<int>(materials.size()) && materialEmits(materials[materialID])) emissivePrimitives.push_back(static_cast<int>(i));
    }
}

void Scene::buildBVH()
{
    bvhNodes.clear();
    bvhPrimitiveIndices.resize(geoms.size());
    for (size_t i = 0; i < geoms.size(); ++i)
    {
        bvhPrimitiveIndices[i] = static_cast<int>(i);
    }
    if (geoms.empty()) return;

    std::vector<Bounds> primitiveBounds(geoms.size());
    std::vector<glm::vec3> primitiveCentroids(geoms.size());
    for (size_t i = 0; i < geoms.size(); ++i)
    {
        primitiveBounds[i] = boundsForGeom(geoms[i]);
        primitiveCentroids[i] = 0.5f * (primitiveBounds[i].minimum + primitiveBounds[i].maximum);
    }

    const auto makeLeaf = [&](int nodeIndex, int begin, int end, const Bounds& bounds) {
        BVHNode& node = bvhNodes[nodeIndex];
        node.boundsMin = bounds.minimum;
        node.boundsMax = bounds.maximum;
        node.firstPrimitive = begin;
        node.primitiveCount = end - begin;
        node.escapeIndex = nodeIndex + 1;
    };

    std::function<int(int, int)> buildNode = [&](int begin, int end) -> int {
        Bounds nodeBounds;
        Bounds centroidBounds;
        for (int i = begin; i < end; ++i)
        {
            const int primitiveIndex = bvhPrimitiveIndices[i];
            nodeBounds.grow(primitiveBounds[primitiveIndex]);
            centroidBounds.grow(primitiveCentroids[primitiveIndex]);
        }

        const int nodeIndex = static_cast<int>(bvhNodes.size());
        bvhNodes.push_back(BVHNode{});
        const int primitiveCount = end - begin;
        if (primitiveCount <= BVH_LEAF_SIZE)
        {
            makeLeaf(nodeIndex, begin, end, nodeBounds);
            return nodeIndex;
        }

        // Multiplying the usual SAH equation by parent area avoids a divide:
        // leaf = N * parentArea; split = parentArea + NL * AL + NR * AR.
        float bestCost = nodeBounds.surfaceArea() * primitiveCount;
        int bestAxis = -1;
        int bestSplit = -1;

        // TODO (BVH study block): Change BVH_SAH_BINS and observe how the
        // chosen tree and render time change.  This is binned SAH: each
        // candidate cost is parentArea + area(left) * count(left) +
        // area(right) * count(right).
        for (int axis = 0; axis < 3; ++axis)
        {
            const float extent = centroidBounds.maximum[axis] - centroidBounds.minimum[axis];
            if (extent <= 1e-6f) continue;

            BVHBin bins[BVH_SAH_BINS];
            const float inverseExtent = static_cast<float>(BVH_SAH_BINS) / extent;
            for (int i = begin; i < end; ++i)
            {
                const int primitiveIndex = bvhPrimitiveIndices[i];
                const int bin = std::min(BVH_SAH_BINS - 1,
                    static_cast<int>((primitiveCentroids[primitiveIndex][axis] - centroidBounds.minimum[axis]) * inverseExtent));
                ++bins[bin].count;
                bins[bin].bounds.grow(primitiveBounds[primitiveIndex]);
            }

            Bounds leftBounds[BVH_SAH_BINS];
            Bounds rightBounds[BVH_SAH_BINS];
            int leftCounts[BVH_SAH_BINS]{};
            int rightCounts[BVH_SAH_BINS]{};
            Bounds runningLeft;
            Bounds runningRight;
            int leftCount = 0;
            int rightCount = 0;
            for (int bin = 0; bin < BVH_SAH_BINS; ++bin)
            {
                if (bins[bin].count > 0) runningLeft.grow(bins[bin].bounds);
                leftCount += bins[bin].count;
                leftBounds[bin] = runningLeft;
                leftCounts[bin] = leftCount;

                const int reverseBin = BVH_SAH_BINS - 1 - bin;
                if (bins[reverseBin].count > 0) runningRight.grow(bins[reverseBin].bounds);
                rightCount += bins[reverseBin].count;
                rightBounds[reverseBin] = runningRight;
                rightCounts[reverseBin] = rightCount;
            }

            for (int split = 0; split < BVH_SAH_BINS - 1; ++split)
            {
                if (leftCounts[split] == 0 || rightCounts[split + 1] == 0) continue;
                const float splitCost = nodeBounds.surfaceArea() +
                    leftBounds[split].surfaceArea() * leftCounts[split] +
                    rightBounds[split + 1].surfaceArea() * rightCounts[split + 1];
                if (splitCost < bestCost)
                {
                    bestCost = splitCost;
                    bestAxis = axis;
                    bestSplit = split;
                }
            }
        }

        if (bestAxis < 0)
        {
            // All centroids coincide, so no spatial split is meaningful.
            makeLeaf(nodeIndex, begin, end, nodeBounds);
            return nodeIndex;
        }

        const float axisMinimum = centroidBounds.minimum[bestAxis];
        const float inverseExtent = static_cast<float>(BVH_SAH_BINS) /
            (centroidBounds.maximum[bestAxis] - axisMinimum);
        const auto middle = std::partition(
            bvhPrimitiveIndices.begin() + begin,
            bvhPrimitiveIndices.begin() + end,
            [&](int primitiveIndex) {
                const int bin = std::min(BVH_SAH_BINS - 1,
                    static_cast<int>((primitiveCentroids[primitiveIndex][bestAxis] - axisMinimum) * inverseExtent));
                return bin <= bestSplit;
            });
        const int mid = static_cast<int>(middle - bvhPrimitiveIndices.begin());
        if (mid == begin || mid == end)
        {
            // Numerical edge cases should not make an empty child.  A median
            // split preserves a finite, balanced construction in that case.
            const int fallbackMid = begin + primitiveCount / 2;
            std::nth_element(bvhPrimitiveIndices.begin() + begin,
                bvhPrimitiveIndices.begin() + fallbackMid,
                bvhPrimitiveIndices.begin() + end,
                [&](int a, int b) { return primitiveCentroids[a][bestAxis] < primitiveCentroids[b][bestAxis]; });
            const int leftRoot = buildNode(begin, fallbackMid);
            const int rightRoot = buildNode(fallbackMid, end);
            bvhNodes[leftRoot].escapeIndex = rightRoot;
        }
        else
        {
            const int leftRoot = buildNode(begin, mid);
            const int rightRoot = buildNode(mid, end);
            bvhNodes[leftRoot].escapeIndex = rightRoot;
        }

        BVHNode& node = bvhNodes[nodeIndex];
        node.boundsMin = nodeBounds.minimum;
        node.boundsMax = nodeBounds.maximum;
        node.firstPrimitive = -1;
        node.primitiveCount = 0;
        node.escapeIndex = static_cast<int>(bvhNodes.size());
        return nodeIndex;
    };

    buildNode(0, static_cast<int>(bvhPrimitiveIndices.size()));
    cout << "Built SAH BVH with " << bvhNodes.size() << " nodes for " << geoms.size() << " primitives." << endl;


    // debugging statements.
    int leafCount = 0;
    int internalCount = 0;
    int minimumLeafSize = static_cast<int>(geoms.size());
    int maximumLeafSize = 0;
    int totalLeafPrimitives = 0;
    for (const BVHNode& node : bvhNodes)
    {
        if (node.primitiveCount == 0)
        {
            ++internalCount;
            continue;
        }
        ++leafCount;
        minimumLeafSize = std::min(minimumLeafSize, node.primitiveCount);
        maximumLeafSize = std::max(maximumLeafSize, node.primitiveCount);
        totalLeafPrimitives += node.primitiveCount;
    }

    int maximumDepth = 0;
    std::function<void(int, int)> measureDepth = [&](int nodeIndex, int depth) {
        maximumDepth = std::max(maximumDepth, depth);
        const BVHNode& node = bvhNodes[nodeIndex];
        if (node.primitiveCount > 0) return;
        const int leftChild = nodeIndex + 1;
        const int rightChild = bvhNodes[leftChild].escapeIndex;
        measureDepth(leftChild, depth + 1);
        measureDepth(rightChild, depth + 1);
    };
    measureDepth(0, 0);

    const Bounds rootBounds{ bvhNodes[0].boundsMin, bvhNodes[0].boundsMax };
    const float rootArea = rootBounds.surfaceArea();
    std::function<float(int)> estimateSAHCost = [&](int nodeIndex) -> float {
        const BVHNode& node = bvhNodes[nodeIndex];
        if (node.primitiveCount > 0) return static_cast<float>(node.primitiveCount);
        const int leftChild = nodeIndex + 1;
        const int rightChild = bvhNodes[leftChild].escapeIndex;
        const float nodeArea = Bounds{ node.boundsMin, node.boundsMax }.surfaceArea();
        if (nodeArea <= 0.0f) return static_cast<float>(geoms.size());
        const float leftArea = Bounds{ bvhNodes[leftChild].boundsMin, bvhNodes[leftChild].boundsMax }.surfaceArea();
        const float rightArea = Bounds{ bvhNodes[rightChild].boundsMin, bvhNodes[rightChild].boundsMax }.surfaceArea();
        return 1.0f + (leftArea / nodeArea) * estimateSAHCost(leftChild) +
            (rightArea / nodeArea) * estimateSAHCost(rightChild);
    };

    cout << "  internal nodes: " << internalCount << ", leaves: " << leafCount
         << ", maximum depth: " << maximumDepth << endl;
    cout << "  leaf primitives (min/avg/max): " << minimumLeafSize << " / "
         << static_cast<float>(totalLeafPrimitives) / leafCount << " / " << maximumLeafSize << endl;
    cout << "  root bounds: min(" << rootBounds.minimum.x << ", " << rootBounds.minimum.y << ", " << rootBounds.minimum.z
         << ") max(" << rootBounds.maximum.x << ", " << rootBounds.maximum.y << ", " << rootBounds.maximum.z
         << "), surface area: " << rootArea << endl;
    cout << "  estimated SAH traversal cost: " << estimateSAHCost(0)
         << " (linear baseline: " << geoms.size() << " primitive tests)" << endl;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    ifstream input(jsonName);
    json data = json::parse(input);
    unordered_map<string, uint32_t> materialNames;
    for (const auto& item : data["Materials"].items())
    {
        const json& p = item.value();
        Material material = makeDefaultMaterial();
        material.type = parseMaterialType(p.value("TYPE", "Diffuse"));
        const auto& color = p["RGB"];
        material.color = glm::vec3(color[0], color[1], color[2]);
        material.emittance = p.value("EMITTANCE", 0.0f);
        material.emission = material.color * material.emittance;
        material.indexOfRefraction = p.value("IOR", p.value("INDEX_OF_REFRACTION", 1.55f));
        material.metallic = p.value("METALLIC", (material.type == MATERIAL_COOK_TORRANCE || material.type == MATERIAL_MICROFACETS) ? 1.0f : 0.0f);
        const float oldRoughness = material.type == MATERIAL_MICROFACETS ? 0.2f : (material.type == MATERIAL_COOK_TORRANCE ? 0.20f : 1.0f);
        material.roughness = glm::clamp(p.value("ROUGHNESS", oldRoughness), 0.001f, 1.0f);
        materialNames[item.key()] = static_cast<uint32_t>(materials.size());
        materials.push_back(material);
    }
    for (const json& p : data["Objects"])
    {
        Geom geometry{};
        geometry.type = p["TYPE"] == "cube" ? CUBE : SPHERE;
        geometry.materialid = materialNames.at(p["MATERIAL"]);
        const auto& t = p["TRANS"]; const auto& r = p["ROTAT"]; const auto& s = p["SCALE"];
        geometry.translation = glm::vec3(t[0], t[1], t[2]); geometry.rotation = glm::vec3(r[0], r[1], r[2]); geometry.scale = glm::vec3(s[0], s[1], s[2]);
        geometry.transform = utilityCore::buildTransformationMatrix(geometry.translation, geometry.rotation, geometry.scale);
        geometry.inverseTransform = glm::inverse(geometry.transform);
        geometry.invTranspose = glm::inverseTranspose(geometry.transform);
        geoms.push_back(geometry);
    }
    const json& c = data["Camera"];
    Camera& camera = state.camera;
    camera.resolution = glm::ivec2(c["RES"][0], c["RES"][1]);
    state.iterations = c["ITERATIONS"]; state.traceDepth = c["DEPTH"]; state.imageName = c["FILE"];
    camera.position = glm::vec3(c["EYE"][0], c["EYE"][1], c["EYE"][2]);
    camera.lookAt = glm::vec3(c["LOOKAT"][0], c["LOOKAT"][1], c["LOOKAT"][2]);
    camera.up = glm::vec3(c["UP"][0], c["UP"][1], c["UP"][2]);
    finalizeCamera(camera, state, tan(c["FOVY"].get<float>() * PI / 180.0f));
    rebuildEmissivePrimitives();
}

void Scene::loadFromGLTF(const std::string& gltfName)
{
    try
    {
        const filesystem::path inputPath(gltfName);
        json document;
        vector<unsigned char> binaryChunk;
        if (lowercase(inputPath.extension().string()) == ".glb")
        {
            const vector<unsigned char> glb = readBinaryFile(inputPath);
            if (glb.size() < 12 || readLE32(glb, 0) != GLB_MAGIC || readLE32(glb, 4) != 2 || readLE32(glb, 8) > glb.size()) throw runtime_error("invalid GLB 2.0 header");
            size_t offset = 12; string jsonChunk;
            while (offset + 8 <= glb.size())
            {
                const uint32_t length = readLE32(glb, offset), type = readLE32(glb, offset + 4); offset += 8;
                if (offset + length > glb.size()) throw runtime_error("truncated GLB chunk");
                if (type == GLB_JSON_CHUNK) jsonChunk.assign(reinterpret_cast<const char*>(glb.data() + offset), length);
                else if (type == GLB_BIN_CHUNK && binaryChunk.empty()) binaryChunk.assign(glb.begin() + offset, glb.begin() + offset + length);
                offset += length;
            }
            while (!jsonChunk.empty() && jsonChunk.back() == '\0') jsonChunk.pop_back();
            if (jsonChunk.empty()) throw runtime_error("GLB has no JSON chunk");
            document = json::parse(jsonChunk);
        }
        else
        {
            ifstream input(gltfName);
            if (!input) throw runtime_error("could not open glTF file");
            document = json::parse(input);
        }
        if (document.value("asset", json::object()).value("version", "") != "2.0") throw runtime_error("only glTF 2.0 is supported");

        vector<vector<unsigned char>> buffers;
        for (size_t i = 0; i < document.value("buffers", json::array()).size(); ++i)
        {
            const json& definition = document.at("buffers").at(i); vector<unsigned char> buffer;
            if (definition.contains("uri"))
            {
                const string uri = definition.at("uri").get<string>();
                if (uri.rfind("data:", 0) == 0)
                {
                    const size_t comma = uri.find(',');
                    if (comma == string::npos || uri.find(";base64") == string::npos) throw runtime_error("only base64 data URIs are supported");
                    buffer = decodeBase64(uri.substr(comma + 1));
                }
                else buffer = readBinaryFile(inputPath.parent_path() / filesystem::path(uri));
            }
            else if (i == 0 && !binaryChunk.empty()) buffer = binaryChunk;
            else throw runtime_error("buffer has no URI or GLB binary chunk");
            if (buffer.size() < definition.value("byteLength", size_t(0))) throw runtime_error("buffer is shorter than declared");
            buffers.push_back(move(buffer));
        }

        // Decode glTF texture sources before materials reference them.  Both
        // external/data-URI images and GLB buffer-view images are supported.
        const json imageDefinitions = document.value("images", json::array());
        const json textureDefinitions = document.value("textures", json::array());
        const json samplerDefinitions = document.value("samplers", json::array());
        vector<int> gltfTextureToSceneTexture(textureDefinitions.size(), -1);
        for (size_t textureIndex = 0; textureIndex < textureDefinitions.size(); ++textureIndex)
        {
            const json& textureDefinition = textureDefinitions.at(textureIndex);
            if (!textureDefinition.contains("source"))
            {
                cerr << "Warning: glTF texture " << textureIndex << " has no supported image source" << endl;
                continue;
            }
            const int imageIndex = textureDefinition.at("source").get<int>();
            if (imageIndex < 0 || imageIndex >= static_cast<int>(imageDefinitions.size()))
                throw runtime_error("texture image index out of range");
            const json& imageDefinition = imageDefinitions.at(imageIndex);
            vector<unsigned char> ownedImageBytes;
            const unsigned char* imageBytes = nullptr;
            size_t imageByteCount = 0;
            if (imageDefinition.contains("uri"))
            {
                const string uri = imageDefinition.at("uri").get<string>();
                if (uri.rfind("data:", 0) == 0)
                {
                    const size_t comma = uri.find(',');
                    if (comma == string::npos || uri.find(";base64") == string::npos)
                        throw runtime_error("only base64 image data URIs are supported");
                    ownedImageBytes = decodeBase64(uri.substr(comma + 1));
                }
                else
                {
                    ownedImageBytes = readBinaryFile(inputPath.parent_path() / filesystem::path(uri));
                }
                imageBytes = ownedImageBytes.data();
                imageByteCount = ownedImageBytes.size();
            }
            else if (imageDefinition.contains("bufferView"))
            {
                const int viewIndex = imageDefinition.at("bufferView").get<int>();
                const json& views = document.value("bufferViews", json::array());
                if (viewIndex < 0 || viewIndex >= static_cast<int>(views.size()))
                    throw runtime_error("image bufferView index out of range");
                const json& view = views.at(viewIndex);
                const int bufferIndex = view.at("buffer").get<int>();
                if (bufferIndex < 0 || bufferIndex >= static_cast<int>(buffers.size()))
                    throw runtime_error("image bufferView buffer index out of range");
                const size_t offset = view.value("byteOffset", size_t(0));
                imageByteCount = view.at("byteLength").get<size_t>();
                if (offset > buffers[bufferIndex].size() || imageByteCount > buffers[bufferIndex].size() - offset)
                    throw runtime_error("image bufferView exceeds its buffer");
                imageBytes = buffers[bufferIndex].data() + offset;
            }
            else
            {
                cerr << "Warning: glTF image " << imageIndex << " has no URI or bufferView" << endl;
                continue;
            }
            if (imageByteCount == 0 || imageByteCount > static_cast<size_t>(numeric_limits<int>::max()))
                throw runtime_error("image is empty or too large to decode");

            int width = 0, height = 0, channels = 0;
            unsigned char* decoded = stbi_load_from_memory(imageBytes, static_cast<int>(imageByteCount),
                &width, &height, &channels, 4);
            if (decoded == nullptr || width <= 0 || height <= 0)
            {
                cerr << "Warning: could not decode glTF texture " << textureIndex << ": "
                     << (stbi_failure_reason() ? stbi_failure_reason() : "unknown image error") << endl;
                if (decoded) stbi_image_free(decoded);
                continue;
            }

            TextureInfo texture{};
            texture.width = width;
            texture.height = height;
            texture.texelOffset = static_cast<int>(textureTexels.size());
            // glTF defaults: REPEAT wrapping and implementation-defined
            // filtering.  Store explicit sampler values when they are present.
            texture.wrapS = 10497;
            texture.wrapT = 10497;
            texture.minFilter = -1;
            texture.magFilter = -1;
            if (textureDefinition.contains("sampler"))
            {
                const int samplerIndex = textureDefinition.at("sampler").get<int>();
                if (samplerIndex < 0 || samplerIndex >= static_cast<int>(samplerDefinitions.size()))
                    throw runtime_error("texture sampler index out of range");
                const json& sampler = samplerDefinitions.at(samplerIndex);
                texture.wrapS = sampler.value("wrapS", texture.wrapS);
                texture.wrapT = sampler.value("wrapT", texture.wrapT);
                texture.minFilter = sampler.value("minFilter", texture.minFilter);
                texture.magFilter = sampler.value("magFilter", texture.magFilter);
            }
            const size_t texelCount = static_cast<size_t>(width) * height;
            textureTexels.reserve(textureTexels.size() + texelCount);
            for (size_t texel = 0; texel < texelCount; ++texel)
            {
                const unsigned char* rgba = decoded + texel * 4;
                textureTexels.push_back(glm::vec4(rgba[0], rgba[1], rgba[2], rgba[3]) / 255.0f);
            }
            stbi_image_free(decoded);
            gltfTextureToSceneTexture[textureIndex] = static_cast<int>(textures.size());
            textures.push_back(texture);
        }

        const auto resolveTexture = [&](const json& textureInfo, const char* usage) -> int {
            const int textureIndex = textureInfo.value("index", -1);
            if (textureIndex < 0 || textureIndex >= static_cast<int>(gltfTextureToSceneTexture.size()))
                throw runtime_error(string(usage) + " texture index out of range");
            if (textureInfo.value("texCoord", 0) != 0)
            {
                cerr << "Warning: " << usage << " uses TEXCOORD_1+, which is not supported" << endl;
                return -1;
            }
            if (textureInfo.contains("extensions") && textureInfo.at("extensions").contains("KHR_texture_transform"))
                cerr << "Warning: " << usage << " KHR_texture_transform is not yet supported" << endl;
            return gltfTextureToSceneTexture[textureIndex];
        };

        vector<int> materialIDs;
        for (const json& definition : document.value("materials", json::array()))
        {
            Material material = makeDefaultMaterial(); material.type = MATERIAL_COOK_TORRANCE;
            const json pbr = definition.value("pbrMetallicRoughness", json::object());
            if (pbr.contains("baseColorFactor")) { const auto& f = pbr.at("baseColorFactor"); material.color = glm::vec3(f.at(0).get<float>(), f.at(1).get<float>(), f.at(2).get<float>()); }
            material.metallic = glm::clamp(pbr.value("metallicFactor", 1.0f), 0.0f, 1.0f);
            material.roughness = glm::clamp(pbr.value("roughnessFactor", 1.0f), 0.001f, 1.0f);
            material.indexOfRefraction = definition.value("ior", 1.5f);
            if (definition.contains("emissiveFactor")) { const auto& f = definition.at("emissiveFactor"); material.emission = glm::vec3(f.at(0).get<float>(), f.at(1).get<float>(), f.at(2).get<float>()); }
            const json extensions = definition.value("extensions", json::object());
            // pbrt_to_gltf stores physical area-light radiance in extras while
            // emissiveFactor only preserves its normalized display color.
            // Prefer the former when it is available so imported PBRT Cornell
            // scenes retain their intended lighting intensity.
            const json extras = definition.value("extras", json::object());
            const json pbrt = extras.value("pbrt", json::object());
            const bool isPbrtDielectric = pbrt.is_object() &&
                lowercase(pbrt.value("type", string())) == "dielectric";
            bool hasPbrtAreaLightRadiance = false;
            if (pbrt.is_object() && pbrt.contains("area_light_radiance_rgb"))
            {
                const json& radiance = pbrt.at("area_light_radiance_rgb");
                if (radiance.is_array() && radiance.size() >= 3)
                {
                    material.emission = glm::vec3(radiance.at(0).get<float>(),
                        radiance.at(1).get<float>(), radiance.at(2).get<float>());
                    hasPbrtAreaLightRadiance = true;
                }
                else
                {
                    cerr << "Warning: ignoring malformed pbrt area-light radiance for material " << materialIDs.size() << endl;
                }
            }
            if (isPbrtDielectric && pbrt.contains("eta"))
            {
                const float eta = pbrt.value("eta", material.indexOfRefraction);
                if (eta > 0.0f) material.indexOfRefraction = eta;
                else cerr << "Warning: ignoring non-positive PBRT eta for material " << materialIDs.size() << endl;
            }
            // Standard glTF material extensions take precedence when both
            // formats provide the same property.
            if (extensions.contains("KHR_materials_ior")) material.indexOfRefraction = extensions.at("KHR_materials_ior").value("ior", material.indexOfRefraction);
            if (!hasPbrtAreaLightRadiance && extensions.contains("KHR_materials_emissive_strength")) material.emission *= extensions.at("KHR_materials_emissive_strength").value("emissiveStrength", 1.0f);
            if (maxComponent(material.emission) > 0.0f) material.type = MATERIAL_EMISSIVE;
            else if (isPbrtDielectric ||
                (extensions.contains("KHR_materials_transmission") && extensions.at("KHR_materials_transmission").value("transmissionFactor", 0.0f) > 0.0f))
            {
                material.type = MATERIAL_DIELECTRIC;
                cout << "Imported dielectric material '" << definition.value("name", "<unnamed>")
                     << "' with IOR " << material.indexOfRefraction << endl;
            }
            if (pbr.contains("baseColorTexture"))
                material.baseColorTexture = resolveTexture(pbr.at("baseColorTexture"), "baseColorTexture");
            if (definition.contains("normalTexture"))
            {
                const json& normalTexture = definition.at("normalTexture");
                material.normalTexture = resolveTexture(normalTexture, "normalTexture");
                material.normalScale = normalTexture.value("scale", 1.0f);
            }
            materialIDs.push_back(static_cast<int>(materials.size())); materials.push_back(material);
        }
        const int defaultMaterialID = static_cast<int>(materials.size());
        Material defaultMaterial = makeDefaultMaterial(); defaultMaterial.type = MATERIAL_COOK_TORRANCE; materials.push_back(defaultMaterial);

        bool hasBounds = false; glm::vec3 boundsMin(0.0f), boundsMax(0.0f);
        const json meshes = document.value("meshes", json::array());
        auto importMesh = [&](int meshIndex, const glm::mat4& world) {
            if (meshIndex < 0 || meshIndex >= static_cast<int>(meshes.size())) throw runtime_error("node mesh index out of range");
            const json primitives = meshes.at(meshIndex).value("primitives", json::array());
            for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex)
            {
                try
                {
                    const json& primitive = primitives.at(primitiveIndex);
                    if (primitive.value("mode", GLTF_MODE_TRIANGLES) != GLTF_MODE_TRIANGLES) { cerr << "Warning: skipping non-triangle glTF primitive" << endl; continue; }
                    const json& attributes = primitive.at("attributes");
                    if (!attributes.contains("POSITION")) throw runtime_error("primitive has no POSITION");
                    const vector<glm::vec3> positions = readVec3(document, buffers, attributes.at("POSITION").get<int>());
                    const bool hasNormals = attributes.contains("NORMAL");
                    const vector<glm::vec3> normals = hasNormals ? readVec3(document, buffers, attributes.at("NORMAL").get<int>()) : vector<glm::vec3>();
                    if (hasNormals && normals.size() != positions.size()) throw runtime_error("NORMAL and POSITION counts differ");
                    const bool hasTextureCoordinates = attributes.contains("TEXCOORD_0");
                    const vector<glm::vec2> textureCoordinates = hasTextureCoordinates ?
                        readVec2(document, buffers, attributes.at("TEXCOORD_0").get<int>()) : vector<glm::vec2>();
                    if (hasTextureCoordinates && textureCoordinates.size() != positions.size())
                        throw runtime_error("TEXCOORD_0 and POSITION counts differ");
                    vector<uint32_t> indices = primitive.contains("indices") ? readIndices(document, buffers, primitive.at("indices").get<int>()) : vector<uint32_t>();
                    if (!primitive.contains("indices")) { indices.resize(positions.size()); for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<uint32_t>(i); }
                    if (indices.size() % 3) throw runtime_error("triangle index count is not divisible by three");
                    int materialID = defaultMaterialID;
                    if (primitive.contains("material")) { const int sourceID = primitive.at("material").get<int>(); if (sourceID < 0 || sourceID >= static_cast<int>(materialIDs.size())) throw runtime_error("material index out of range"); materialID = materialIDs[sourceID]; }
                    const Material& primitiveMaterial = materials[materialID];
                    if (!hasTextureCoordinates && (primitiveMaterial.baseColorTexture >= 0 || primitiveMaterial.normalTexture >= 0))
                        cerr << "Warning: textured glTF primitive has no TEXCOORD_0; its texture maps will be skipped" << endl;
                    const glm::mat3 normalMatrix = hasNormals ? glm::transpose(glm::inverse(glm::mat3(world))) : glm::mat3(1.0f);
                    for (size_t i = 0; i < indices.size(); i += 3)
                    {
                        Geom triangle{}; triangle.type = TRIANGLE; triangle.materialid = materialID;
                        triangle.transform = triangle.inverseTransform = triangle.invTranspose = glm::mat4(1.0f);
                        triangle.hasVertexNormals = hasNormals ? 1 : 0;
                        triangle.hasTextureCoordinates = hasTextureCoordinates ? 1 : 0;
                        for (int vertex = 0; vertex < 3; ++vertex)
                        {
                            const uint32_t positionIndex = indices[i + vertex]; if (positionIndex >= positions.size()) throw runtime_error("triangle index out of range");
                            triangle.triangleVertices[vertex] = glm::vec3(world * glm::vec4(positions[positionIndex], 1.0f));
                            if (hasNormals) triangle.triangleNormals[vertex] = glm::normalize(normalMatrix * normals[positionIndex]);
                            if (hasTextureCoordinates) triangle.triangleUVs[vertex] = textureCoordinates[positionIndex];
                            if (!hasBounds) { boundsMin = boundsMax = triangle.triangleVertices[vertex]; hasBounds = true; }
                            else { boundsMin = glm::min(boundsMin, triangle.triangleVertices[vertex]); boundsMax = glm::max(boundsMax, triangle.triangleVertices[vertex]); }
                        }
                        geoms.push_back(triangle);
                    }
                }
                catch (const exception& error) { cerr << "Warning: skipping glTF primitive " << primitiveIndex << " in mesh " << meshIndex << ": " << error.what() << endl; }
            }
        };

        const json nodes = document.value("nodes", json::array()), scenes = document.value("scenes", json::array());
        int cameraIndex = -1; glm::mat4 cameraTransform(1.0f);
        function<void(int, const glm::mat4&)> visitNode;
        visitNode = [&](int nodeIndex, const glm::mat4& parent) {
            if (nodeIndex < 0 || nodeIndex >= static_cast<int>(nodes.size())) throw runtime_error("node index out of range");
            const json& node = nodes.at(nodeIndex); const glm::mat4 world = parent * getNodeTransform(node);
            if (node.contains("mesh")) importMesh(node.at("mesh").get<int>(), world);
            if (cameraIndex < 0 && node.contains("camera")) { cameraIndex = node.at("camera").get<int>(); cameraTransform = world; }
            for (const json& child : node.value("children", json::array())) visitNode(child.get<int>(), world);
        };
        vector<int> roots;
        const int sceneIndex = document.value("scene", scenes.empty() ? -1 : 0);
        if (sceneIndex >= 0 && sceneIndex < static_cast<int>(scenes.size())) for (const json& node : scenes.at(sceneIndex).value("nodes", json::array())) roots.push_back(node.get<int>());
        else
        {
            vector<bool> child(nodes.size(), false);
            for (const json& node : nodes) for (const json& value : node.value("children", json::array())) { const int i = value.get<int>(); if (i >= 0 && i < static_cast<int>(child.size())) child[i] = true; }
            for (size_t i = 0; i < child.size(); ++i) if (!child[i]) roots.push_back(static_cast<int>(i));
        }
        for (int root : roots) visitNode(root, glm::mat4(1.0f));
        if (geoms.empty()) throw runtime_error("scene contains no supported triangle geometry");

        Camera& camera = state.camera; camera.resolution = glm::ivec2(800, 800); state.iterations = 1000; state.traceDepth = 8; state.imageName = inputPath.stem().string();
        float yscaled = tan(45.0f * PI / 180.0f);
        const json cameras = document.value("cameras", json::array());
        if (cameraIndex >= 0 && cameraIndex < static_cast<int>(cameras.size()) && cameras.at(cameraIndex).value("type", "") == "perspective")
        {
            const json& p = cameras.at(cameraIndex).at("perspective"); yscaled = tan(0.5f * p.at("yfov").get<float>());
            camera.position = glm::vec3(cameraTransform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            camera.lookAt = camera.position + glm::normalize(glm::vec3(cameraTransform * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
            camera.up = glm::normalize(glm::vec3(cameraTransform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
        }
        else
        {
            const glm::vec3 center = 0.5f * (boundsMin + boundsMax); const float radius = std::max(0.5f * glm::length(boundsMax - boundsMin), 0.5f);
            camera.position = center + glm::vec3(0.0f, 0.0f, 2.5f * radius); camera.lookAt = center; camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
        }
        finalizeCamera(camera, state, yscaled);
        rebuildEmissivePrimitives();
    }
    catch (const exception& error) { cerr << "Failed to load glTF scene '" << gltfName << "': " << error.what() << endl; exit(-1); }
}

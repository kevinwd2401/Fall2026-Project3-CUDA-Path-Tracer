#include "scene.h"

#include "utilities.h"
#include "pathtrace_config.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "json.hpp"
#include <stb_image.h>

// Compile the pinned C++ tinygltf implementation in this host translation unit.
// Texture decoding stays deferred so only used maps consume the texel budget.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tiny_gltf.h>

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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace
{
constexpr size_t FILE_READ_CHUNK_SIZE = 64ull * 1024ull * 1024ull;
// Keep all imported material textures within a predictable host/device memory
// budget. Small scenes retain their original resolution; large collections
// are uniformly reduced before being stored as packed RGBA8.
constexpr size_t GLTF_TEXTURE_TEXEL_BUDGET = 64ull * 1024ull * 1024ull;

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

Bounds boundsForCube(const Cube& cube)
{
    Bounds bounds;
    for (int corner = 0; corner < 8; ++corner)
    {
        const glm::vec3 local(
            (corner & 1) ? 0.5f : -0.5f,
            (corner & 2) ? 0.5f : -0.5f,
            (corner & 4) ? 0.5f : -0.5f);
        bounds.grow(glm::vec3(cube.transform * glm::vec4(local, 1.0f)));
    }
    return bounds;
}

Bounds boundsForSphere(const Sphere& sphere)
{
    // A transformed sphere fits within the transformed unit-cube bounds.
    Cube cube{};
    cube.transform = sphere.transform;
    return boundsForCube(cube);
}

Bounds boundsForTriangle(const Triangle& triangle)
{
    Bounds bounds;
    bounds.grow(triangle.triangleVertices[0]);
    bounds.grow(triangle.triangleVertices[1]);
    bounds.grow(triangle.triangleVertices[2]);
    return bounds;
}

Bounds boundsForPrimitive(const PrimitiveRef& primitive, const std::vector<Cube>& cubes,
    const std::vector<Sphere>& spheres, const std::vector<Triangle>& triangles, const Volume& volume)
{
    switch (primitive.type)
    {
    case CUBE: return boundsForCube(cubes[primitive.index]);
    case SPHERE: return boundsForSphere(spheres[primitive.index]);
    case TRIANGLE: return boundsForTriangle(triangles[primitive.index]);
    case VOLUME: return Bounds{volume.boundsMin, volume.boundsMax};
    }
    return Bounds{};
}

int materialForPrimitive(const PrimitiveRef& primitive, const std::vector<Cube>& cubes,
    const std::vector<Sphere>& spheres, const std::vector<Triangle>& triangles)
{
    switch (primitive.type)
    {
    case CUBE: return cubes[primitive.index].materialid;
    case SPHERE: return spheres[primitive.index].materialid;
    case TRIANGLE: return triangles[primitive.index].materialid;
    case VOLUME: return -1; // medium parameters are independent of surface materials
    }
    return -1;
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
    material.alpha = 1.0f;
    material.emission = glm::vec3(0.0f);
    material.type = MATERIAL_DIFFUSE;
    material.indexOfRefraction = 1.55f;
    material.roughness = 1.0f;
    material.baseColorTexture = -1;
    material.normalTexture = -1;
    material.metallicRoughnessTexture = -1;
    material.normalScale = 1.0f;
    material.alphaMode = ALPHA_OPAQUE;
    material.alphaCutoff = 0.5f;
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

    input.seekg(0, ios::end);
    const streampos end = input.tellg();
    const streamoff byteCount = static_cast<streamoff>(end);
    if (byteCount < 0 || static_cast<uintmax_t>(byteCount) > numeric_limits<size_t>::max())
        throw runtime_error("could not determine the size of " + path.string());
    input.seekg(0, ios::beg);

    vector<unsigned char> bytes(static_cast<size_t>(byteCount));
    size_t offset = 0;
    while (offset < bytes.size())
    {
        const size_t count = min(bytes.size() - offset, FILE_READ_CHUNK_SIZE);
        input.read(reinterpret_cast<char*>(bytes.data() + offset), static_cast<streamsize>(count));
        if (!input) throw runtime_error("could not read " + path.string());
        offset += count;
    }
    return bytes;
}

// Keep chunked reads for large assets while tinygltf owns container/URI parsing.
bool readGLTFFile(vector<unsigned char>* output, string* error,
    const string& filename, void*)
{
    try
    {
        *output = readBinaryFile(filesystem::path(filename));
        return true;
    }
    catch (const exception& failure)
    {
        if (error) *error = failure.what();
        return false;
    }
}

// Data-URI bytes must survive parsing. Buffer-view images can borrow the model's
// buffers later; external images are read on demand (TINYGLTF_NO_EXTERNAL_IMAGE).
bool retainEncodedGLTFImage(tinygltf::Image* image, int, string* error, string*,
    int, int, const unsigned char* bytes, int size, void*)
{
    image->as_is = true;
    if (image->bufferView >= 0) return true;
    if (size <= 0 || bytes == nullptr)
    {
        if (error) *error += "embedded image is empty or too large to decode";
        return false;
    }
    image->image.assign(bytes, bytes + size);
    return true;
}

const tinygltf::Value& gltfMember(const tinygltf::Value& object, const string& key)
{
    static const tinygltf::Value absent;
    return object.IsObject() ? object.Get(key) : absent;
}

float gltfNumber(const tinygltf::Value& object, const string& key, float fallback)
{
    if (!object.Has(key)) return fallback;
    const tinygltf::Value& value = object.Get(key);
    if (!value.IsNumber()) throw runtime_error(key + " must be a number");
    return static_cast<float>(value.GetNumberAsDouble());
}

struct AccessorView
{
    const vector<unsigned char>* buffer;
    size_t offset, stride, count;
    int componentType, components;
};

AccessorView getAccessor(const tinygltf::Model& model, int accessorIndex)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
        throw runtime_error("accessor index out of range");
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    if (accessor.sparse.isSparse || accessor.bufferView < 0)
        throw runtime_error("sparse or bufferless accessors are unsupported");
    if (accessor.bufferView >= static_cast<int>(model.bufferViews.size()))
        throw runtime_error("buffer view index out of range");
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
        throw runtime_error("buffer index out of range");
    AccessorView result{};
    result.buffer = &model.buffers[view.buffer].data;
    result.componentType = accessor.componentType;
    result.components = tinygltf::GetNumComponentsInType(accessor.type);
    result.count = accessor.count;
    const int componentBytes = tinygltf::GetComponentSizeInBytes(result.componentType);
    if (componentBytes <= 0 || result.components <= 0)
        throw runtime_error("unsupported accessor type");
    const size_t packedSize = static_cast<size_t>(componentBytes) * result.components;
    result.stride = view.byteStride ? view.byteStride : packedSize;
    const size_t viewOffset = view.byteOffset;
    const size_t viewLength = view.byteLength;
    const size_t accessorOffset = accessor.byteOffset;
    if (viewOffset > result.buffer->size() || viewLength > result.buffer->size() - viewOffset)
        throw runtime_error("buffer view exceeds its buffer");
    if (accessorOffset > viewLength) throw runtime_error("accessor starts outside its buffer view");
    result.offset = viewOffset + accessorOffset;
    const size_t available = viewLength - accessorOffset;
    if (result.stride < packedSize) throw runtime_error("accessor stride is smaller than its element");
    if (result.count > 0)
    {
        if (packedSize > available ||
            (result.count > 1 && result.count - 1 > (available - packedSize) / result.stride))
            throw runtime_error("accessor exceeds its buffer view");
    }
    return result;
}

struct EncodedImage
{
    vector<unsigned char> ownedBytes;
    const unsigned char* borrowedBytes = nullptr;
    size_t byteCount = 0;

    const unsigned char* data() const
    {
        return ownedBytes.empty() ? borrowedBytes : ownedBytes.data();
    }
};

void appendTextureImage(vector<uchar4>& destination, const unsigned char* source,
    int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
{
    const size_t oldSize = destination.size();
    const size_t targetTexelCount = static_cast<size_t>(targetWidth) * targetHeight;
    if (targetTexelCount > numeric_limits<size_t>::max() - oldSize)
        throw runtime_error("texture texel count is too large");
    destination.resize(oldSize + targetTexelCount);

    uchar4* output = destination.data() + oldSize;
    if (sourceWidth == targetWidth && sourceHeight == targetHeight)
    {
        static_assert(sizeof(uchar4) == 4, "uchar4 must contain four packed bytes");
        memcpy(output, source, targetTexelCount * sizeof(uchar4));
        return;
    }

    // A center sample is enough here because device-side linear filtering is
    // still applied. More importantly, work is proportional to the retained
    // texture data rather than every source pixel in a multi-gigabyte set.
    for (int y = 0; y < targetHeight; ++y)
    {
        const int sourceY = min(sourceHeight - 1,
            static_cast<int>((static_cast<int64_t>(2 * y + 1) * sourceHeight) / (2ll * targetHeight)));
        for (int x = 0; x < targetWidth; ++x)
        {
            const int sourceX = min(sourceWidth - 1,
                static_cast<int>((static_cast<int64_t>(2 * x + 1) * sourceWidth) / (2ll * targetWidth)));
            const unsigned char* rgba = source + (static_cast<size_t>(sourceY) * sourceWidth + sourceX) * 4;
            output[static_cast<size_t>(y) * targetWidth + x] = make_uchar4(rgba[0], rgba[1], rgba[2], rgba[3]);
        }
    }
}

vector<glm::vec3> readVec3(const tinygltf::Model& model, int accessorIndex)
{
    const AccessorView view = getAccessor(model, accessorIndex);
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

vector<glm::vec2> readVec2(const tinygltf::Model& model, int accessorIndex)
{
    const AccessorView view = getAccessor(model, accessorIndex);
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

vector<uint32_t> readIndices(const tinygltf::Model& model, int accessorIndex)
{
    const AccessorView view = getAccessor(model, accessorIndex);
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

glm::mat4 getNodeTransform(const tinygltf::Node& node)
{
    if (!node.matrix.empty())
    {
        const auto& values = node.matrix;
        if (values.size() != 16) throw runtime_error("node matrix must have 16 values");
        glm::mat4 matrix(1.0f);
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                matrix[column][row] = static_cast<float>(values.at(column * 4 + row));
        return matrix;
    }
    glm::vec3 translation(0.0f), scale(1.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (!node.translation.empty())
    {
        const auto& v = node.translation;
        translation = glm::vec3(v.at(0), v.at(1), v.at(2));
    }
    if (!node.scale.empty())
    {
        const auto& v = node.scale;
        scale = glm::vec3(v.at(0), v.at(1), v.at(2));
    }
    if (!node.rotation.empty())
    {
        const auto& v = node.rotation;
        rotation = glm::quat(v.at(3), v.at(0), v.at(1), v.at(2));
    }
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}
}

Scene::Scene(string filename, string environmentFilename, string volumeFilename, float volumeScale)
{
    if (filename.empty())
    {
        if (environmentFilename.empty() || volumeFilename.empty())
            throw runtime_error("A volume-only scene requires both an HDRI and a NanoVDB file");
        cout << "Creating volume-only scene ..." << endl;
    }
    else
    {
        cout << "Reading scene from " << filename << " ..." << endl << " " << endl;
        const string extension = lowercase(filesystem::path(filename).extension().string());
        if (extension == ".json") loadFromJSON(filename);
        else if (extension == ".gltf" || extension == ".glb") loadFromGLTF(filename);
        else throw runtime_error("Unsupported scene file: " + filename);
    }
    if (!environmentFilename.empty()) loadEnvironmentMap(environmentFilename);
    if (!volumeFilename.empty())
    {
        volume = loadNanoVDB(volumeFilename, volumeScale);
        primitives.push_back({VOLUME, 0});
        cout << "Loaded NanoVDB density " << volumeFilename << " (scale " << volumeScale << ")" << endl;
    }
    if (filename.empty())
    {
        Camera& camera = state.camera;
        camera.resolution = glm::ivec2(1200, 800);
        state.iterations = 6000;
        state.traceDepth = 8;
        state.imageName = filesystem::path(volumeFilename).stem().string();
        camera.lookAt = volume.medium.scaleCenter;
        camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
        // Fit the scaled bounds along -Z with a full vertical FOV of 60 degrees.
        // At the front face, both horizontal and vertical extents fit with margin.
        const float yscaled = tan(30.0f * PI / 180.0f);
        const float xscaled = yscaled * float(camera.resolution.x) / float(camera.resolution.y);
        const glm::vec3 halfExtent = volume.medium.boundsMax * 0.5f - volume.medium.boundsMin * 0.5f;
        const float distance = std::max(0.001f, halfExtent.z +
            1.1f * std::max(halfExtent.x / xscaled, halfExtent.y / yscaled));
        camera.position = camera.lookAt + glm::vec3(0.0f, 0.0f, distance);
        if (!std::isfinite(distance) || !std::isfinite(camera.position.z) ||
            !(camera.position.z > volume.medium.boundsMax.z))
            throw runtime_error("Volume bounds are too extreme to frame a default camera");
        finalizeCamera(camera, state, yscaled);
    }
#if USE_BVH
    buildBVH();
#endif
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
    environment.aliasProbability.resize(pixelCount);
    environment.aliasIndex.resize(pixelCount);
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

    for (size_t index = 0; index < pixelCount; ++index)
    {
        const int y = static_cast<int>(index / width);
        const float theta0 = PI * static_cast<float>(y) / static_cast<float>(height);
        const float theta1 = PI * static_cast<float>(y + 1) / static_cast<float>(height);
        const float texelSolidAngle = deltaPhi * (cosf(theta0) - cosf(theta1));
        environment.pdfSolidAngle[index] = texelSolidAngle > 0.0f ?
            weights[index] / static_cast<float>(totalWeight) / texelSolidAngle : 0.0f;
    }

    // Walker's alias method stores one primary texel and one fallback texel
    // per table entry.  It exactly represents the normalized texel weights
    // while allowing O(1) sampling in the CUDA kernel.
    vector<float> scaledProbability(pixelCount);
    vector<size_t> small;
    vector<size_t> large;
    small.reserve(pixelCount);
    large.reserve(pixelCount);
    for (size_t index = 0; index < pixelCount; ++index)
    {
        scaledProbability[index] = static_cast<float>(weights[index] * pixelCount / totalWeight);
        if (scaledProbability[index] < 1.0f) small.push_back(index);
        else large.push_back(index);
    }
    while (!small.empty() && !large.empty())
    {
        const size_t low = small.back();
        small.pop_back();
        const size_t high = large.back();
        large.pop_back();
        environment.aliasProbability[low] = scaledProbability[low];
        environment.aliasIndex[low] = static_cast<int>(high);
        scaledProbability[high] = scaledProbability[high] + scaledProbability[low] - 1.0f;
        if (scaledProbability[high] < 1.0f) small.push_back(high);
        else large.push_back(high);
    }
    for (size_t index : small)
    {
        environment.aliasProbability[index] = 1.0f;
        environment.aliasIndex[index] = static_cast<int>(index);
    }
    for (size_t index : large)
    {
        environment.aliasProbability[index] = 1.0f;
        environment.aliasIndex[index] = static_cast<int>(index);
    }
    cout << "Loaded HDRI environment map " << filename << " (" << width << "x" << height
         << ", importance sampled)." << endl;
}

void Scene::rebuildEmissivePrimitives()
{
    emissivePrimitives.clear();
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        const int materialID = materialForPrimitive(primitives[i], cubes, spheres, triangles);
        if (materialID >= 0 && materialID < static_cast<int>(materials.size()) && materialEmits(materials[materialID])) emissivePrimitives.push_back(static_cast<int>(i));
    }
}

void Scene::buildBVH()
{
    bvhNodes.clear();
    bvhPrimitiveIndices.resize(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        bvhPrimitiveIndices[i] = static_cast<int>(i);
    }
    if (primitives.empty()) return;

    std::vector<Bounds> primitiveBounds(primitives.size());
    std::vector<glm::vec3> primitiveCentroids(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        primitiveBounds[i] = boundsForPrimitive(primitives[i], cubes, spheres, triangles, volume.medium);
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

        // Binned SAH candidate cost: parentArea + area(left) * count(left)
        // + area(right) * count(right).
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
    cout << "Built SAH BVH with " << bvhNodes.size() << " nodes for " << primitives.size() << " primitives." << endl;


    // debugging statements.
    int leafCount = 0;
    int internalCount = 0;
    int minimumLeafSize = static_cast<int>(primitives.size());
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
        if (nodeArea <= 0.0f) return static_cast<float>(primitives.size());
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
         << " (linear baseline: " << primitives.size() << " primitive tests)" << endl;
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
        const GeomType type = p["TYPE"] == "cube" ? CUBE : SPHERE;
        const int materialid = materialNames.at(p["MATERIAL"]);
        const auto& t = p["TRANS"]; const auto& r = p["ROTAT"]; const auto& s = p["SCALE"];
        const glm::mat4 transform = utilityCore::buildTransformationMatrix(
            glm::vec3(t[0], t[1], t[2]), glm::vec3(r[0], r[1], r[2]), glm::vec3(s[0], s[1], s[2]));
        if (type == CUBE)
        {
            Cube cube{};
            cube.materialid = materialid;
            cube.transform = transform;
            cube.inverseTransform = glm::inverse(transform);
            cube.invTranspose = glm::inverseTranspose(transform);
            primitives.push_back({ CUBE, static_cast<int>(cubes.size()) });
            cubes.push_back(cube);
        }
        else
        {
            Sphere sphere{};
            sphere.materialid = materialid;
            sphere.transform = transform;
            sphere.inverseTransform = glm::inverse(transform);
            sphere.invTranspose = glm::inverseTranspose(transform);
            primitives.push_back({ SPHERE, static_cast<int>(spheres.size()) });
            spheres.push_back(sphere);
        }
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
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        string error, warning;
        const tinygltf::FsCallbacks filesystemCallbacks{
            tinygltf::FileExists, tinygltf::ExpandFilePath, readGLTFFile,
            tinygltf::WriteWholeFile, tinygltf::GetFileSizeInBytes, nullptr
        };
        if (!loader.SetFsCallbacks(filesystemCallbacks, &error)) throw runtime_error(error);
        loader.SetMaxExternalFileSize(numeric_limits<size_t>::max());
        loader.SetImageLoader(retainEncodedGLTFImage, nullptr);
        const uintmax_t fileSize = filesystem::file_size(inputPath);
        // The pinned tinygltf in-memory entry points use unsigned-int lengths.
        if (fileSize == 0 || fileSize > numeric_limits<unsigned int>::max())
            throw runtime_error("glTF/GLB file is empty or too large to parse");
        const bool loaded = lowercase(inputPath.extension().string()) == ".glb" ?
            loader.LoadBinaryFromFile(&model, &error, &warning, gltfName) :
            loader.LoadASCIIFromFile(&model, &error, &warning, gltfName);
        if (!warning.empty()) cerr << "Warning: " << warning << endl;
        if (!loaded) throw runtime_error(error.empty() ? "could not parse glTF scene" : error);
        if (!error.empty()) cerr << "Warning: " << error << endl;
        if (model.asset.version != "2.0") throw runtime_error("only glTF 2.0 is supported");

        // Decode only the texture channels the renderer uses. Images shared by
        // textures with different samplers share one texel allocation.
        const auto& imageDefinitions = model.images;
        const auto& textureDefinitions = model.textures;
        const auto& samplerDefinitions = model.samplers;
        const auto& materialDefinitions = model.materials;

        vector<bool> requiredTextures(textureDefinitions.size(), false);
        const auto requireTexture = [&](int textureIndex, const char* usage) {
            if (textureIndex == -1) return;
            if (textureIndex < 0 || textureIndex >= static_cast<int>(requiredTextures.size()))
                throw runtime_error(string(usage) + " texture index out of range");
            requiredTextures[textureIndex] = true;
        };
        for (const tinygltf::Material& definition : materialDefinitions)
        {
            const auto& pbr = definition.pbrMetallicRoughness;
            requireTexture(pbr.baseColorTexture.index, "baseColorTexture");
            requireTexture(pbr.metallicRoughnessTexture.index, "metallicRoughnessTexture");
            requireTexture(definition.normalTexture.index, "normalTexture");
        }

        const auto getEncodedImage = [&](int imageIndex) -> EncodedImage {
            if (imageIndex < 0 || imageIndex >= static_cast<int>(imageDefinitions.size()))
                throw runtime_error("texture image index out of range");
            const tinygltf::Image& image = imageDefinitions[imageIndex];
            EncodedImage result;
            if (!image.uri.empty())
            {
                string decodedUri;
                if (!tinygltf::URIDecode(image.uri, &decodedUri, nullptr))
                    throw runtime_error("invalid image URI");
                result.ownedBytes = readBinaryFile(inputPath.parent_path() / filesystem::path(decodedUri));
                result.byteCount = result.ownedBytes.size();
            }
            else if (image.bufferView >= 0)
            {
                if (image.bufferView >= static_cast<int>(model.bufferViews.size()))
                    throw runtime_error("image bufferView index out of range");
                const tinygltf::BufferView& view = model.bufferViews[image.bufferView];
                if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
                    throw runtime_error("image bufferView buffer index out of range");
                const auto& bytes = model.buffers[view.buffer].data;
                result.byteCount = view.byteLength;
                if (view.byteOffset > bytes.size() || result.byteCount > bytes.size() - view.byteOffset)
                    throw runtime_error("image bufferView exceeds its buffer");
                result.borrowedBytes = bytes.data() + view.byteOffset;
            }
            else
            {
                result.borrowedBytes = image.image.data();
                result.byteCount = image.image.size();
            }

            if (result.byteCount == 0 || result.byteCount > static_cast<size_t>(numeric_limits<int>::max()))
                throw runtime_error("image is empty or too large to decode");
            return result;
        };

        struct ImageSize { int sourceWidth = 0, sourceHeight = 0, targetWidth = 0, targetHeight = 0; };
        vector<ImageSize> imageSizes(imageDefinitions.size());
        unordered_set<int> requiredImages;
        for (size_t textureIndex = 0; textureIndex < textureDefinitions.size(); ++textureIndex)
        {
            if (!requiredTextures[textureIndex]) continue;
            const tinygltf::Texture& textureDefinition = textureDefinitions.at(textureIndex);
            if (textureDefinition.source == -1)
            {
                cerr << "Warning: glTF texture " << textureIndex << " has no supported image source" << endl;
                continue;
            }
            const int imageIndex = textureDefinition.source;
            if (imageIndex < 0 || imageIndex >= static_cast<int>(imageDefinitions.size()))
                throw runtime_error("texture image index out of range");
            requiredImages.insert(imageIndex);
        }

        size_t sourceTexelCount = 0;
        for (int imageIndex : requiredImages)
        {
            const EncodedImage encoded = getEncodedImage(imageIndex);
            int width = 0, height = 0, channels = 0;
            if (!stbi_info_from_memory(encoded.data(), static_cast<int>(encoded.byteCount), &width, &height, &channels) ||
                width <= 0 || height <= 0)
            {
                cerr << "Warning: could not inspect glTF image " << imageIndex << ": "
                     << (stbi_failure_reason() ? stbi_failure_reason() : "unknown image error") << endl;
                continue;
            }
            if (static_cast<size_t>(width) > numeric_limits<size_t>::max() / static_cast<size_t>(height))
                throw runtime_error("texture dimensions are too large");
            const size_t imageTexels = static_cast<size_t>(width) * height;
            if (imageTexels > numeric_limits<size_t>::max() - sourceTexelCount)
                throw runtime_error("texture texel count is too large");
            sourceTexelCount += imageTexels;
            imageSizes[imageIndex].sourceWidth = width;
            imageSizes[imageIndex].sourceHeight = height;
        }

        const double textureScale = sourceTexelCount > GLTF_TEXTURE_TEXEL_BUDGET ?
            sqrt(static_cast<double>(GLTF_TEXTURE_TEXEL_BUDGET) / static_cast<double>(sourceTexelCount)) : 1.0;
        size_t retainedTexelCount = 0;
        for (int imageIndex : requiredImages)
        {
            ImageSize& size = imageSizes[imageIndex];
            if (size.sourceWidth <= 0 || size.sourceHeight <= 0) continue;
            size.targetWidth = max(1, min(size.sourceWidth, static_cast<int>(floor(size.sourceWidth * textureScale))));
            size.targetHeight = max(1, min(size.sourceHeight, static_cast<int>(floor(size.sourceHeight * textureScale))));
            const size_t imageTexels = static_cast<size_t>(size.targetWidth) * size.targetHeight;
            if (imageTexels > static_cast<size_t>(numeric_limits<int>::max()) - retainedTexelCount)
                throw runtime_error("packed texture offsets exceed their 32-bit representation");
            retainedTexelCount += imageTexels;
        }
        textureTexels.reserve(retainedTexelCount);
        textures.reserve(count(requiredTextures.begin(), requiredTextures.end(), true));
        if (textureScale < 1.0)
        {
            cout << "Texture set contains " << sourceTexelCount << " texels; resizing to approximately "
                 << retainedTexelCount << " texels (" << retainedTexelCount * sizeof(uchar4) / (1024 * 1024)
                 << " MiB packed RGBA8)." << endl;
        }

        struct StoredImage { int width = 0, height = 0, texelOffset = -1; };
        vector<StoredImage> storedImages(imageDefinitions.size());
        vector<int> gltfTextureToSceneTexture(textureDefinitions.size(), -1);
        for (size_t textureIndex = 0; textureIndex < textureDefinitions.size(); ++textureIndex)
        {
            if (!requiredTextures[textureIndex]) continue;
            const tinygltf::Texture& textureDefinition = textureDefinitions.at(textureIndex);
            if (textureDefinition.source == -1) continue;
            const int imageIndex = textureDefinition.source;
            StoredImage& stored = storedImages.at(imageIndex);
            if (stored.texelOffset < 0)
            {
                const ImageSize& requestedSize = imageSizes.at(imageIndex);
                if (requestedSize.targetWidth <= 0 || requestedSize.targetHeight <= 0) continue;
                const EncodedImage encoded = getEncodedImage(imageIndex);
                int width = 0, height = 0, channels = 0;
                unique_ptr<unsigned char, decltype(&stbi_image_free)> decoded(
                    stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.byteCount),
                        &width, &height, &channels, 4), stbi_image_free);
                if (!decoded || width <= 0 || height <= 0)
                {
                    cerr << "Warning: could not decode glTF image " << imageIndex << ": "
                         << (stbi_failure_reason() ? stbi_failure_reason() : "unknown image error") << endl;
                    continue;
                }
                const int targetWidth = max(1, min(width, requestedSize.targetWidth));
                const int targetHeight = max(1, min(height, requestedSize.targetHeight));
                stored.texelOffset = static_cast<int>(textureTexels.size());
                stored.width = targetWidth;
                stored.height = targetHeight;
                appendTextureImage(textureTexels, decoded.get(), width, height, targetWidth, targetHeight);
            }

            TextureInfo texture{};
            texture.width = stored.width;
            texture.height = stored.height;
            texture.texelOffset = stored.texelOffset;
            // glTF defaults: REPEAT wrapping and implementation-defined
            // filtering. Store explicit sampler values when they are present.
            texture.wrapS = 10497;
            texture.wrapT = 10497;
            texture.minFilter = -1;
            texture.magFilter = -1;
            if (textureDefinition.sampler != -1)
            {
                const int samplerIndex = textureDefinition.sampler;
                if (samplerIndex < 0 || samplerIndex >= static_cast<int>(samplerDefinitions.size()))
                    throw runtime_error("texture sampler index out of range");
                const tinygltf::Sampler& sampler = samplerDefinitions.at(samplerIndex);
                texture.wrapS = sampler.wrapS;
                texture.wrapT = sampler.wrapT;
                texture.minFilter = sampler.minFilter;
                texture.magFilter = sampler.magFilter;
            }
            gltfTextureToSceneTexture[textureIndex] = static_cast<int>(textures.size());
            textures.push_back(texture);
        }
        if (!textures.empty())
        {
            cout << "Loaded " << textures.size() << " material textures using "
                 << textureTexels.size() * sizeof(uchar4) / (1024 * 1024) << " MiB." << endl;
        }

        // TextureInfo and NormalTextureInfo have the same lookup fields.
        const auto resolveTexture = [&](const auto& textureInfo, const char* usage) -> int {
            const int textureIndex = textureInfo.index;
            if (textureIndex == -1) return -1;
            if (textureIndex < 0 || textureIndex >= static_cast<int>(gltfTextureToSceneTexture.size()))
                throw runtime_error(string(usage) + " texture index out of range");
            if (textureInfo.texCoord != 0)
            {
                cerr << "Warning: " << usage << " uses TEXCOORD_1+, which is not supported" << endl;
                return -1;
            }
            if (textureInfo.extensions.count("KHR_texture_transform"))
                cerr << "Warning: " << usage << " KHR_texture_transform is not yet supported" << endl;
            return gltfTextureToSceneTexture[textureIndex];
        };

        vector<int> materialIDs;
        for (const tinygltf::Material& definition : materialDefinitions)
        {
            Material material = makeDefaultMaterial(); material.type = MATERIAL_COOK_TORRANCE;
            const auto& pbr = definition.pbrMetallicRoughness;
            const auto& f = pbr.baseColorFactor;
            material.color = glm::vec3(f.at(0), f.at(1), f.at(2));
            material.alpha = glm::clamp(static_cast<float>(f.at(3)), 0.0f, 1.0f);
            if (definition.alphaMode == "MASK") material.alphaMode = ALPHA_MASK;
            else if (definition.alphaMode == "BLEND") material.alphaMode = ALPHA_BLEND;
            material.alphaCutoff = glm::clamp(static_cast<float>(definition.alphaCutoff), 0.0f, 1.0f);
            material.metallic = glm::clamp(static_cast<float>(pbr.metallicFactor), 0.0f, 1.0f);
            material.roughness = glm::clamp(static_cast<float>(pbr.roughnessFactor), 0.001f, 1.0f);
            // Preserve the legacy, nonstandard material-level "ior" property too.
            const auto legacyIOR = definition.additionalValues.find("ior");
            material.indexOfRefraction = legacyIOR == definition.additionalValues.end() ?
                1.5f : static_cast<float>(legacyIOR->second.number_value);
            const auto& emission = definition.emissiveFactor;
            material.emission = glm::vec3(emission.at(0), emission.at(1), emission.at(2));
            const auto& extensions = definition.extensions;
            // pbrt_to_gltf stores physical area-light radiance in extras while
            // emissiveFactor only preserves its normalized display color.
            const tinygltf::Value& pbrt = gltfMember(definition.extras, "pbrt");
            const tinygltf::Value& pbrtType = gltfMember(pbrt, "type");
            const bool isPbrtDielectric = pbrtType.IsString() &&
                lowercase(pbrtType.Get<string>()) == "dielectric";
            bool hasPbrtAreaLightRadiance = false;
            if (pbrt.Has("area_light_radiance_rgb"))
            {
                const tinygltf::Value& radiance = pbrt.Get("area_light_radiance_rgb");
                if (radiance.IsArray() && radiance.ArrayLen() >= 3 &&
                    radiance.Get(0).IsNumber() && radiance.Get(1).IsNumber() && radiance.Get(2).IsNumber())
                {
                    material.emission = glm::vec3(radiance.Get(0).GetNumberAsDouble(),
                        radiance.Get(1).GetNumberAsDouble(), radiance.Get(2).GetNumberAsDouble());
                    hasPbrtAreaLightRadiance = true;
                }
                else
                {
                    cerr << "Warning: ignoring malformed pbrt area-light radiance for material " << materialIDs.size() << endl;
                }
            }
            if (isPbrtDielectric && pbrt.Has("eta"))
            {
                const float eta = gltfNumber(pbrt, "eta", material.indexOfRefraction);
                if (eta > 0.0f) material.indexOfRefraction = eta;
                else cerr << "Warning: ignoring non-positive PBRT eta for material " << materialIDs.size() << endl;
            }
            // Standard glTF extensions retain their existing precedence.
            if (extensions.count("KHR_materials_ior"))
                material.indexOfRefraction = gltfNumber(extensions.at("KHR_materials_ior"), "ior", material.indexOfRefraction);
            if (!hasPbrtAreaLightRadiance && extensions.count("KHR_materials_emissive_strength"))
                material.emission *= gltfNumber(extensions.at("KHR_materials_emissive_strength"), "emissiveStrength", 1.0f);
            if (maxComponent(material.emission) > 0.0f) material.type = MATERIAL_EMISSIVE;
            else if (isPbrtDielectric ||
                (extensions.count("KHR_materials_transmission") &&
                    gltfNumber(extensions.at("KHR_materials_transmission"), "transmissionFactor", 0.0f) > 0.0f))
            {
                material.type = MATERIAL_DIELECTRIC;
                cout << "Imported dielectric material '" << (definition.name.empty() ? "<unnamed>" : definition.name)
                     << "' with IOR " << material.indexOfRefraction << endl;
            }
            material.baseColorTexture = resolveTexture(pbr.baseColorTexture, "baseColorTexture");
            material.metallicRoughnessTexture = resolveTexture(pbr.metallicRoughnessTexture, "metallicRoughnessTexture");
            material.normalTexture = resolveTexture(definition.normalTexture, "normalTexture");
            material.normalScale = static_cast<float>(definition.normalTexture.scale);
            materialIDs.push_back(static_cast<int>(materials.size())); materials.push_back(material);
        }
        const int defaultMaterialID = static_cast<int>(materials.size());
        Material defaultMaterial = makeDefaultMaterial(); defaultMaterial.type = MATERIAL_COOK_TORRANCE; materials.push_back(defaultMaterial);

        bool hasBounds = false; glm::vec3 boundsMin(0.0f), boundsMax(0.0f);
        const auto& meshes = model.meshes;
        const auto& nodes = model.nodes;
        const auto& scenes = model.scenes;

        // Avoid repeated reallocations while expanding indexed meshes into the
        // flat triangle representation. Account for mesh instancing when the
        // same mesh is referenced by more than one node.
        vector<size_t> meshInstanceCounts(meshes.size(), 0);
        for (const tinygltf::Node& node : nodes)
        {
            if (node.mesh == -1) continue;
            const int meshIndex = node.mesh;
            if (meshIndex < 0 || meshIndex >= static_cast<int>(meshes.size()))
                throw runtime_error("node mesh index out of range");
            ++meshInstanceCounts[meshIndex];
        }
        size_t estimatedTriangleCount = 0;
        const auto& accessors = model.accessors;
        for (size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
        {
            if (meshInstanceCounts[meshIndex] == 0) continue;
            const tinygltf::Mesh& mesh = meshes.at(meshIndex);
            const auto& meshPrimitives = mesh.primitives;
            size_t meshTriangleCount = 0;
            for (const tinygltf::Primitive& primitive : meshPrimitives)
            {
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) continue;
                int accessorIndex = primitive.indices;
                const auto position = primitive.attributes.find("POSITION");
                if (accessorIndex == -1 && position != primitive.attributes.end())
                    accessorIndex = position->second;
                if (accessorIndex < 0 || accessorIndex >= static_cast<int>(accessors.size())) continue;
                meshTriangleCount += accessors.at(accessorIndex).count / 3;
            }
            if (meshTriangleCount > (numeric_limits<size_t>::max() - estimatedTriangleCount) /
                meshInstanceCounts[meshIndex]) throw runtime_error("scene has too many triangles");
            estimatedTriangleCount += meshTriangleCount * meshInstanceCounts[meshIndex];
        }
        if (estimatedTriangleCount > static_cast<size_t>(numeric_limits<int>::max()) - triangles.size())
            throw runtime_error("scene has too many triangles for 32-bit primitive indices");
        triangles.reserve(triangles.size() + estimatedTriangleCount);
        primitives.reserve(primitives.size() + estimatedTriangleCount);

        auto importMesh = [&](int meshIndex, const glm::mat4& world) {
            if (meshIndex < 0 || meshIndex >= static_cast<int>(meshes.size())) throw runtime_error("node mesh index out of range");
            const tinygltf::Mesh& mesh = meshes.at(meshIndex);
            const auto& meshPrimitives = mesh.primitives;
            for (size_t primitiveIndex = 0; primitiveIndex < meshPrimitives.size(); ++primitiveIndex)
            {
                try
                {
                    const tinygltf::Primitive& primitive = meshPrimitives.at(primitiveIndex);
                    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) { cerr << "Warning: skipping non-triangle glTF primitive" << endl; continue; }
                    const auto& attributes = primitive.attributes;
                    if (attributes.count("POSITION") == 0) throw runtime_error("primitive has no POSITION");
                    const vector<glm::vec3> positions = readVec3(model, attributes.at("POSITION"));
                    const bool hasNormals = (attributes.count("NORMAL") != 0);
                    const vector<glm::vec3> normals = hasNormals ? readVec3(model, attributes.at("NORMAL")) : vector<glm::vec3>();
                    if (hasNormals && normals.size() != positions.size()) throw runtime_error("NORMAL and POSITION counts differ");
                    const bool hasTextureCoordinates = (attributes.count("TEXCOORD_0") != 0);
                    const vector<glm::vec2> textureCoordinates = hasTextureCoordinates ?
                        readVec2(model, attributes.at("TEXCOORD_0")) : vector<glm::vec2>();
                    if (hasTextureCoordinates && textureCoordinates.size() != positions.size())
                        throw runtime_error("TEXCOORD_0 and POSITION counts differ");
                    vector<uint32_t> indices = primitive.indices != -1 ? readIndices(model, primitive.indices) : vector<uint32_t>();
                    if (primitive.indices == -1) { indices.resize(positions.size()); for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<uint32_t>(i); }
                    if (indices.size() % 3) throw runtime_error("triangle index count is not divisible by three");
                    int materialID = defaultMaterialID;
                    if (primitive.material != -1) { const int sourceID = primitive.material; if (sourceID < 0 || sourceID >= static_cast<int>(materialIDs.size())) throw runtime_error("material index out of range"); materialID = materialIDs[sourceID]; }
                    const Material& primitiveMaterial = materials[materialID];
                    if (!hasTextureCoordinates && (primitiveMaterial.baseColorTexture >= 0 ||
                        primitiveMaterial.normalTexture >= 0 || primitiveMaterial.metallicRoughnessTexture >= 0))
                        cerr << "Warning: textured glTF primitive has no TEXCOORD_0; its texture maps will be skipped" << endl;
                    const glm::mat3 normalMatrix = hasNormals ? glm::transpose(glm::inverse(glm::mat3(world))) : glm::mat3(1.0f);
                    for (size_t i = 0; i < indices.size(); i += 3)
                    {
                        Triangle triangle{}; triangle.materialid = materialID;
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
                        primitives.push_back({ TRIANGLE, static_cast<int>(triangles.size()) });
                        triangles.push_back(triangle);
                    }
                }
                catch (const exception& error) { cerr << "Warning: skipping glTF primitive " << primitiveIndex << " in mesh " << meshIndex << ": " << error.what() << endl; }
            }
        };

        int cameraIndex = -1; glm::mat4 cameraTransform(1.0f);
        vector<unsigned char> activeNodes(nodes.size(), 0);
        function<void(int, const glm::mat4&)> visitNode;
        visitNode = [&](int nodeIndex, const glm::mat4& parent) {
            if (nodeIndex < 0 || nodeIndex >= static_cast<int>(nodes.size())) throw runtime_error("node index out of range");
            if (activeNodes[nodeIndex]) throw runtime_error("cycle in glTF node hierarchy");
            activeNodes[nodeIndex] = 1;
            const tinygltf::Node& node = nodes.at(nodeIndex); const glm::mat4 world = parent * getNodeTransform(node);
            if (node.mesh != -1) importMesh(node.mesh, world);
            if (cameraIndex < 0 && node.camera != -1) { cameraIndex = node.camera; cameraTransform = world; }
            for (int child : node.children) visitNode(child, world);
            activeNodes[nodeIndex] = 0;
        };
        vector<int> roots;
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : (scenes.empty() ? -1 : 0);
        if (sceneIndex >= 0 && sceneIndex < static_cast<int>(scenes.size())) roots = scenes.at(sceneIndex).nodes;
        else
        {
            vector<bool> child(nodes.size(), false);
            for (const tinygltf::Node& node : nodes) for (int i : node.children) { if (i >= 0 && i < static_cast<int>(child.size())) child[i] = true; }
            for (size_t i = 0; i < child.size(); ++i) if (!child[i]) roots.push_back(static_cast<int>(i));
        }
        for (int root : roots) visitNode(root, glm::mat4(1.0f));
        if (primitives.empty()) throw runtime_error("scene contains no supported triangle geometry");
        cout << "Imported " << triangles.size() << " glTF triangles." << endl;

        Camera& camera = state.camera; camera.resolution = glm::ivec2(1200, 800); state.iterations = 6000; state.traceDepth = 8; state.imageName = inputPath.stem().string();
        float yscaled = tan(45.0f * PI / 180.0f);
        const auto& cameras = model.cameras;
        if (cameraIndex >= 0 && cameraIndex < static_cast<int>(cameras.size()) && cameras.at(cameraIndex).type == "perspective")
        {
            const auto& p = cameras.at(cameraIndex).perspective; yscaled = tan(0.5f * static_cast<float>(p.yfov));
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

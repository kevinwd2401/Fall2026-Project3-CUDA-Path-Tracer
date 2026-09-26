#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
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

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl << " " << endl;
    const string extension = lowercase(filesystem::path(filename).extension().string());
    if (extension == ".json") loadFromJSON(filename);
    else if (extension == ".gltf" || extension == ".glb") loadFromGLTF(filename);
    else { cout << "Couldn't read from " << filename << endl; exit(-1); }
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
        const float oldRoughness = material.type == MATERIAL_MICROFACETS ? 0.45f : (material.type == MATERIAL_COOK_TORRANCE ? 0.20f : 1.0f);
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
            bool hasPbrtAreaLightRadiance = false;
            if (extras.contains("pbrt") && extras.at("pbrt").contains("area_light_radiance_rgb"))
            {
                const json& radiance = extras.at("pbrt").at("area_light_radiance_rgb");
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
            if (extensions.contains("KHR_materials_ior")) material.indexOfRefraction = extensions.at("KHR_materials_ior").value("ior", material.indexOfRefraction);
            if (!hasPbrtAreaLightRadiance && extensions.contains("KHR_materials_emissive_strength")) material.emission *= extensions.at("KHR_materials_emissive_strength").value("emissiveStrength", 1.0f);
            if (maxComponent(material.emission) > 0.0f) material.type = MATERIAL_EMISSIVE;
            else if (extensions.contains("KHR_materials_transmission") && extensions.at("KHR_materials_transmission").value("transmissionFactor", 0.0f) > 0.0f) material.type = MATERIAL_DIELECTRIC;
            if (pbr.contains("baseColorTexture")) cerr << "Warning: glTF baseColorTexture is ignored for material " << materialIDs.size() << endl;
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
                    vector<uint32_t> indices = primitive.contains("indices") ? readIndices(document, buffers, primitive.at("indices").get<int>()) : vector<uint32_t>();
                    if (!primitive.contains("indices")) { indices.resize(positions.size()); for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<uint32_t>(i); }
                    if (indices.size() % 3) throw runtime_error("triangle index count is not divisible by three");
                    int materialID = defaultMaterialID;
                    if (primitive.contains("material")) { const int sourceID = primitive.at("material").get<int>(); if (sourceID < 0 || sourceID >= static_cast<int>(materialIDs.size())) throw runtime_error("material index out of range"); materialID = materialIDs[sourceID]; }
                    const glm::mat3 normalMatrix = hasNormals ? glm::transpose(glm::inverse(glm::mat3(world))) : glm::mat3(1.0f);
                    for (size_t i = 0; i < indices.size(); i += 3)
                    {
                        Geom triangle{}; triangle.type = TRIANGLE; triangle.materialid = materialID;
                        triangle.transform = triangle.inverseTransform = triangle.invTranspose = glm::mat4(1.0f); triangle.hasVertexNormals = hasNormals ? 1 : 0;
                        for (int vertex = 0; vertex < 3; ++vertex)
                        {
                            const uint32_t positionIndex = indices[i + vertex]; if (positionIndex >= positions.size()) throw runtime_error("triangle index out of range");
                            triangle.triangleVertices[vertex] = glm::vec3(world * glm::vec4(positions[positionIndex], 1.0f));
                            if (hasNormals) triangle.triangleNormals[vertex] = glm::normalize(normalMatrix * normals[positionIndex]);
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

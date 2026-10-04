#pragma once

#include "sceneStructs.h"
#include "volume.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void loadEnvironmentMap(const std::string& filename);
    void rebuildEmissivePrimitives();
    void buildBVH();
public:
    // An empty filename creates a volume-only scene; HDRI and NanoVDB are required.
    Scene(std::string filename, std::string environmentFilename = "", std::string volumeFilename = "",
        float volumeScale = 1.0f);

    // Scene-wide primitive ordering.  Other per-primitive structures store
    // indices into this array; the ref selects an entry in a typed list.
    std::vector<PrimitiveRef> primitives;
    std::vector<Cube> cubes;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    VolumeAsset volume; // exactly one optional grid; owns aligned host storage
    std::vector<BVHNode> bvhNodes;
    std::vector<int> bvhPrimitiveIndices;
    std::vector<Material> materials;
    std::vector<int> emissivePrimitives;
    // glTF texture descriptors and packed RGBA8 texels.
    std::vector<TextureInfo> textures;
    std::vector<uchar4> textureTexels;
    // An equirectangular, linear HDR image, its directional PDF, and an
    // O(1) alias table used to importance-sample its texels on the device.
    struct EnvironmentMap
    {
        int width = 0;
        int height = 0;
        std::vector<glm::vec3> texels;
        std::vector<float> aliasProbability;
        std::vector<int> aliasIndex;
        std::vector<float> pdfSolidAngle;

        bool valid() const
        {
            return width > 0 && height > 0 &&
                texels.size() == static_cast<size_t>(width) * height &&
                aliasProbability.size() == texels.size() &&
                aliasIndex.size() == texels.size() && pdfSolidAngle.size() == texels.size();
        }
    } environment;
    RenderState state;
};

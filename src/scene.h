#pragma once

#include "sceneStructs.h"
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
    Scene(std::string filename, std::string environmentFilename = "");

    std::vector<Geom> geoms;
    std::vector<BVHNode> bvhNodes;
    std::vector<int> bvhPrimitiveIndices;
    std::vector<Material> materials;
    std::vector<int> emissivePrimitives;
    // glTF texture descriptors and their RGBA texels.
    std::vector<TextureInfo> textures;
    std::vector<glm::vec4> textureTexels;
    // An equirectangular, linear HDR image and the precomputed probability
    // distribution used to sample it as a directional light.
    struct EnvironmentMap
    {
        int width = 0;
        int height = 0;
        std::vector<glm::vec3> texels;
        std::vector<float> cdf;
        std::vector<float> pdfSolidAngle;

        bool valid() const
        {
            return width > 0 && height > 0 &&
                texels.size() == static_cast<size_t>(width) * height &&
                cdf.size() == texels.size() && pdfSolidAngle.size() == texels.size();
        }
    } environment;
    RenderState state;
};

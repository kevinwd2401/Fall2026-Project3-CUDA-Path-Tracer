#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void rebuildEmissivePrimitives();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<int> emissivePrimitives;
    RenderState state;
};

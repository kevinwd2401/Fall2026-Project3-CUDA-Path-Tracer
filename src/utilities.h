#pragma once

#include "glm/glm.hpp"

#include <string>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class GuiDataContainer
{
public:
    GuiDataContainer()
        : TracedDepth(0), FocalLength(2.0f), LensRadius(0.008f) {}

    int TracedDepth;
    float FocalLength;
    float LensRadius;
};

namespace utilityCore
{
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
}

#pragma once

#include <cmath>

#include <glm/glm.hpp>

#ifdef __CUDACC__
#define PATHTRACER_HOST_DEVICE __host__ __device__
#else
#define PATHTRACER_HOST_DEVICE
#endif

// Maps scene-linear radiance to display-ready sRGB using the common ACES
// fitted filmic curve. Keep this shared by the CUDA display and PNG export
// paths so an exported image matches the interactive preview.
PATHTRACER_HOST_DEVICE inline glm::vec3 acesFilmicTonemap(const glm::vec3& color)
{
    const glm::vec3 nonNegative = glm::max(color, glm::vec3(0.0f));
    const glm::vec3 mapped =
        (nonNegative * (2.51f * nonNegative + 0.03f)) /
        (nonNegative * (2.43f * nonNegative + 0.59f) + 0.14f);
    const glm::vec3 clamped = glm::clamp(mapped, glm::vec3(0.0f), glm::vec3(1.0f));

    // Gamma correction
    return glm::pow(clamped, glm::vec3(1.0f / 2.2f));
}

#undef PATHTRACER_HOST_DEVICE

#include "volume.h"
#include <nanovdb/io/IO.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

VolumeAsset loadNanoVDB(const std::string& filename, float scale)
{
    VolumeAsset asset;
    if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(1.0f / scale))
        throw std::runtime_error("Volume scale must be positive and finite with a finite reciprocal");
    asset.medium.inverseScale = 1.0f / scale;
    // A file may contain several fields; select its first grid deliberately.
    // Export a float fog-density grid first, with zero background and statistics.
    asset.handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename, 0);
    const auto* grid = asset.handle.grid<float>();
    if (!grid || grid->gridClass() != nanovdb::GridClass::FogVolume)
        throw std::runtime_error("NanoVDB input must have a float FogVolume as its first grid");
    if (!grid->hasBBox() || !grid->hasMinMax() || grid->activeVoxelCount() == 0)
        throw std::runtime_error("NanoVDB density needs a nonempty bounding box and min/max statistics");
    const auto& root = grid->tree().root();
    if (root.background() != 0.0f || !std::isfinite(root.minimum()) ||
        !std::isfinite(root.maximum()) || root.minimum() < 0.0f || root.maximum() < 0.0f)
        throw std::runtime_error("NanoVDB density must be finite and nonnegative, with zero background");
    if (!std::isfinite(asset.medium.extinction) || asset.medium.extinction < 0.0f ||
        !std::isfinite(asset.medium.g) || std::abs(asset.medium.g) >= 1.0f)
        throw std::runtime_error("Volume extinction must be nonnegative; phase g must be in (-1, 1)");
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(asset.medium.albedo[axis]) || asset.medium.albedo[axis] < 0.0f || asset.medium.albedo[axis] > 1.0f)
            throw std::runtime_error("Volume scattering albedo must be in [0, 1]");

    // Define the medium on the active index domain with a one-voxel halo for
    // trilinear reconstruction. Outside this explicit domain the medium is vacuum.
    // Transform all corners, so rotations and nonuniform voxel sizes are supported.
    const auto& bbox = grid->indexBBox();
    glm::vec3 lower(std::numeric_limits<float>::max());
    glm::vec3 upper(-std::numeric_limits<float>::max());
    for (int corner = 0; corner < 8; ++corner)
    {
        const nanovdb::Vec3d index(
            (corner & 1) ? double(bbox.max()[0]) + 1.0 : double(bbox.min()[0]) - 1.0,
            (corner & 2) ? double(bbox.max()[1]) + 1.0 : double(bbox.min()[1]) - 1.0,
            (corner & 4) ? double(bbox.max()[2]) + 1.0 : double(bbox.min()[2]) - 1.0);
        const auto world = grid->indexToWorld(index);
        const glm::vec3 p{float(world[0]), float(world[1]), float(world[2])};
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(p[axis])) throw std::runtime_error("Nonfinite NanoVDB transform/bounds");
        lower = glm::min(lower, p);
        upper = glm::max(upper, p);
    }
    // Scale about the original bounds center, retaining the grid's placement.
    asset.medium.scaleCenter = lower * 0.5f + upper * 0.5f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double center = asset.medium.scaleCenter[axis];
        const float scaledLower = float(center + (double(lower[axis]) - center) * scale);
        const float scaledUpper = float(center + (double(upper[axis]) - center) * scale);
        asset.medium.boundsMin[axis] = std::nextafter(scaledLower, -std::numeric_limits<float>::infinity());
        asset.medium.boundsMax[axis] = std::nextafter(scaledUpper, std::numeric_limits<float>::infinity());
        if (!std::isfinite(asset.medium.boundsMin[axis]) || !std::isfinite(asset.medium.boundsMax[axis]) ||
            !(scaledLower < scaledUpper))
            throw std::runtime_error("Volume scale produces overflowing or collapsed bounds");
    }
    // Trilinear interpolation is a convex combination: the maximum of the
    // stored density bounds it. Add slack for floating-point roundoff.
    asset.medium.majorant = root.maximum() * asset.medium.extinction * 1.0001f;
    if (!std::isfinite(asset.medium.majorant))
        throw std::runtime_error("NanoVDB extinction majorant overflow");
    return asset;
}

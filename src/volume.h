#pragma once

#include "sceneStructs.h"
#include <nanovdb/GridHandle.h>

struct VolumeAsset
{
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    Volume medium;
    bool valid() const { return handle.grid<float>() != nullptr; }
};

VolumeAsset loadNanoVDB(const std::string& filename, float scale = 1.0f);

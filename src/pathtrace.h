#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
// Clear accumulated radiance after camera/lens changes; restart iteration at 1.
void pathtraceResetAccumulation();
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
// Downloads the accumulated image only for an explicit CPU-side export.
void pathtraceCopyImageToHost();

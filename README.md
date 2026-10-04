CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

### Volume-only scenes

No JSON/glTF scene is needed when both an HDRI and a NanoVDB volume are supplied:

```text
cis565_path_tracer.exe studio.hdr cloud.nvdb
cis565_path_tracer.exe cloud.nvdb studio.hdr --volume-scale 0.1
```

Files and options may appear in any order, with at most one scene, HDRI and volume.
Without a scene file, both HDRI and volume are required. The renderer creates a
1200x800 camera looking toward the scaled volume's center from +Z, with a 60-degree
vertical field of view, 6000 iterations and a maximum path depth of 8. The volume
remains in the shared BVH; the HDRI provides lighting. Image output uses the volume
filename stem. Existing camera controls remain available.

### Volume scale

Supply a positive uniform scale (with or without a scene file):

```text
cis565_path_tracer.exe scene.gltf studio.hdr cloud.nvdb --volume-scale 0.1
```

The default is `1`. A value of `0.1` makes the volume one-tenth the size in each
dimension, keeping its bounding-box center fixed. The flag can appear before or
after the optional HDR/NanoVDB files. It requires a NanoVDB file and cannot be
repeated. Both BVH bounds and density sampling use the scale. Density/extinction
remain unchanged per world unit, so a smaller volume has less optical thickness.


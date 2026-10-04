# tinygltf

`include/tiny_gltf.h` is vendored from the upstream **v2.9.6** C++ release:
https://github.com/syoyo/tinygltf/blob/v2.9.6/tiny_gltf.h

The MIT license and third-party notices are included in the header. The header
is unmodified. Pinning the C++ API keeps this project on its existing C++17/CUDA
toolchain; no C runtime or configure-time download is required.

`src/scene.cpp` provides the single `TINYGLTF_IMPLEMENTATION` translation unit
and uses the existing nlohmann JSON header. Its custom image callback retains
encoded data-URI images and borrows buffer-view images; external images are
loaded on demand. The renderer still uses its existing stb implementation in
`src/stb.cpp` to decode only required maps to RGBA8, with shared texel storage
and the existing 64-Mi-texel budget. Filesystem callbacks retain chunked reads
and allow external buffers larger than tinygltf's default 2-GiB limit.

tinygltf handles glTF/GLB parsing and buffer/URI loading. Conversion to renderer
triangles, material/extension/PBRT-extras handling, cameras, and BVH construction
remain in `Scene`. The separate custom JSON scene format also remains supported.

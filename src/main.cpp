#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Separator();
    ImGui::Text("Depth of Field");

    bool lensSettingsChanged = false;
    lensSettingsChanged |= ImGui::SliderFloat(
        "Focal Length", &imguiData->FocalLength, 0.1f, 20.0f, "%.3f");
    lensSettingsChanged |= ImGui::SliderFloat(
        "Lens Radius", &imguiData->LensRadius, 0.0f, 0.1f, "%.4f");
    if (lensSettingsChanged)
    {
        camchanged = true;
    }

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.(json|gltf|glb) [HDRI_FILE] [VOLUME_FILE.nvdb] [--volume-scale FACTOR]\n", argv[0]);
        printf("   or: %s HDRI_FILE VOLUME_FILE.nvdb [--volume-scale FACTOR]\n", argv[0]);
        printf("Files and options can be given in any order; only one of each is allowed.\n");
        printf("Volume scale defaults to 1; positive values scale about the volume center.\n");
        return 1;
    }

    std::string sceneFile, environmentFile, volumeFile;
    float volumeScale = 1.0f;
    bool volumeScaleSpecified = false;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--volume-scale")
            {
                if (volumeScaleSpecified) throw std::runtime_error("--volume-scale may only be supplied once");
                if (++i >= argc) throw std::runtime_error("--volume-scale requires a positive finite number");
                const std::string value = argv[i];
                size_t consumed = 0;
                try { volumeScale = std::stof(value, &consumed); }
                catch (const std::exception&) { throw std::runtime_error("--volume-scale requires a positive finite number"); }
                if (consumed != value.size() || !std::isfinite(volumeScale) || volumeScale <= 0.0f ||
                    !std::isfinite(1.0f / volumeScale))
                    throw std::runtime_error("--volume-scale requires a positive finite number with a finite reciprocal");
                volumeScaleSpecified = true;
                continue;
            }
            if (argument.rfind("--", 0) == 0)
                throw std::runtime_error("Unknown option: " + argument);
            std::string extension = std::filesystem::path(argv[i]).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            const bool sceneArgument = extension == ".json" || extension == ".gltf" || extension == ".glb";
            std::string& target = sceneArgument ? sceneFile : (extension == ".nvdb" ? volumeFile : environmentFile);
            if (!target.empty()) throw std::runtime_error("Only one scene, one HDRI and one NanoVDB file may be supplied");
            target = argv[i];
        }
        if (volumeScaleSpecified && volumeFile.empty())
            throw std::runtime_error("--volume-scale requires a NanoVDB file");
        if (sceneFile.empty() && (environmentFile.empty() || volumeFile.empty()))
            throw std::runtime_error("Supply a JSON/glTF/GLB scene, or both an HDRI and a NanoVDB file");
        scene = new Scene(sceneFile, environmentFile, volumeFile, volumeScale);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        delete scene;
        return EXIT_FAILURE;
    }
    if (scene->emissivePrimitives.empty() && !scene->environment.valid())
    {
        std::cerr << "Error: scene has no light sources. Add an emissive primitive or pass an HDRI "
                  << "environment map." << std::endl;
        delete scene;
        return EXIT_FAILURE;
    }

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // Compute spherical orbit coordinates from the camera offset. Using the
    // offset directly avoids normalizing a zero XZ or YZ projection when the
    // initial camera is aligned with a world axis.
    const glm::vec3 cameraOffset = cam.position - cam.lookAt;
    zoom = std::fmax(glm::length(cameraOffset), 0.1f);
    phi = std::atan2(cameraOffset.x, cameraOffset.z);
    const float normalizedHeight = std::fmax(-1.0f, std::fmin(1.0f, cameraOffset.y / zoom));
    theta = std::acos(normalizedHeight);
    theta = std::fmax(0.001f, std::fmin(theta, PI - 0.001f));
    cameraPosition = cameraOffset;
    ogLookAt = cam.lookAt;

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);
    pathtraceInit(scene);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    // The renderer keeps its accumulation buffer on the GPU during
    // interactive rendering.
    pathtraceCopyImageToHost();
    // Avoid dividing by zero when the user saves before the first iteration.
    float samples = static_cast<float>(std::max(iteration, 1));
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        const glm::vec3 v = cam.view;
        const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        cam.right = glm::normalize(glm::cross(v, worldUp));
        cam.up = glm::normalize(glm::cross(cam.right, v));

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceResetAccumulation();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        glm::vec3 movementAxis(0.0f);
        bool movementKey = true;
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        switch (key)
        {
            case GLFW_KEY_W: movementAxis = cam.view; break;
            case GLFW_KEY_S: movementAxis = -cam.view; break;
            case GLFW_KEY_A: movementAxis = -cam.right; break;
            case GLFW_KEY_D: movementAxis = cam.right; break;
            default: movementKey = false; break;
        }

        if (movementKey)
        {
            const glm::vec3 translation = glm::normalize(movementAxis) *
                std::fmax(0.01f, zoom * 0.05f);

            // Translate the camera and its point of interest together. Keeping
            // their offset unchanged preserves the orbit radius and makes the
            // next left-button tumble rotate around the translated target.
            cam.position += translation;
            cam.lookAt += translation;
            camchanged = true;
            return;
        }
    }

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
            case GLFW_KEY_X:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // Do not start a camera drag from an ImGui window, but always process the
    // release so a drag cannot remain latched if the cursor crosses the UI.
    if (MouseOverImGuiWindow() && action == GLFW_PRESS)
    {
        return;
    }

    const bool pressed = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        leftMousePressed = pressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        rightMousePressed = pressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        middleMousePressed = pressed;
    }

    if (pressed)
    {
        // Initialize the drag anchor at press time so the first motion event
        // does not jump from the global (0, 0) cursor position.
        glfwGetCursorPos(window, &lastX, &lastY);
    }
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX && ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // Tumble/orbit around the current point of interest.
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        phi = std::fmod(phi, 2.0f * PI);
        theta = std::fmax(0.001f, std::fmin(theta, PI - 0.001f));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        // Dolly horizontally: dragging right moves away, dragging left moves
        // toward the target. Exponential scaling keeps the control useful at
        // both near and far distances.
        zoom *= std::exp(static_cast<float>((xpos - lastX) / width));
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        // Track/pan in the camera's screen plane. pixelLength.y is the world
        // size of one pixel at a unit distance, so scaling it by zoom keeps
        // pan speed proportional to the current framing.
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        const float panScale = zoom * cam.pixelLength.y;
        const glm::vec3 right = glm::normalize(cam.right);
        const glm::vec3 up = glm::normalize(cam.up);
        cam.lookAt -= static_cast<float>(xpos - lastX) * right * panScale;
        cam.lookAt += static_cast<float>(ypos - lastY) * up * panScale;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    (void) xoffset;
    if (MouseOverImGuiWindow())
    {
        return;
    }

    // Positive wheel motion conventionally means up, which zooms in.
    zoom *= std::pow(0.85f, static_cast<float>(yoffset));
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
}

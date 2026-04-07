#define NOMINMAX
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

constexpr float PI = 3.14159265358979323846f;

#include <ktx.h>
#include <ktxvulkan.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <random>
#include <utilities.hpp>

#define TYPE_S  0x01 // 0b00000001 // (stationary or moving) solid boundary
#define TYPE_E  0x02 // 0b00000010 // equilibrium boundary (inflow/outflow)
#define TYPE_T  0x04 // 0b00000100 // temperature boundary
#define TYPE_F  0x08 // 0b00001000 // fluid
#define TYPE_I  0x10 // 0b00010000 // interface
#define TYPE_G  0x20 // 0b00100000 // gas
#define TYPE_X  0x40 // 0b01000000 // reserved type X
#define TYPE_Y  0x80 // 0b10000000 // reserved type Y

#define TYPE_MS 0x03 // 0b00000011 // cell next to moving solid boundary
#define TYPE_BO 0x03 // 0b00000011 // any flag bit used for boundaries (temperature excluded)
#define TYPE_IF 0x18 // 0b00011000 // change from interface to fluid
#define TYPE_IG 0x30 // 0b00110000 // change from interface to gas
#define TYPE_GI 0x38 // 0b00111000 // change from gas to interface
#define TYPE_SU 0x38 // 0b00111000 // any flag bit used for SURFACE

uint32_t WIDTH = 1280;
uint32_t HEIGHT = 720;
const uint32_t Nx = 128;
const uint32_t Ny = 128;
const uint32_t Nz = 128;
const uint32_t Nxyz = Nx * Ny * Nz;
const uint32_t D = 3;
const uint32_t Q = 19;

const int MAX_FRAMES_IN_FLIGHT = 1;

uint32_t lagrangianPointCount = 0;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_LUNARG_monitor"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_EXT_shader_atomic_float",
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsAndComputeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities = {};
    std::vector<VkSurfaceFormatKHR> formats{};
    std::vector<VkPresentModeKHR> presentModes{};
};

struct vecQ {
    float f[Q];
};

struct SimulateUBO {
    alignas(4) uint32_t Nx = 0;
    alignas(4) uint32_t Ny = 0;
    alignas(4) uint32_t Nz = 0;
    alignas(4) uint32_t Nxyz = 0;
    alignas(4) uint32_t particleCount = 0;
    alignas(4) float particleRho = 1.0f;
    alignas(4) float niu = 0.0f;
    alignas(4) float tau = 0.0f;
    alignas(4) float inv_tau = 0.0f;
    alignas(4) float fx = 0.0f;
    alignas(4) float fy = 0.0f;
    alignas(4) float fz = 0.0f;
    alignas(4) float dt = 0.0f;
    alignas(4) uint32_t t = 0;
    alignas(4) uint32_t render_mode = 0;
    alignas(4) uint32_t fixed_point_iteration = 0;
    alignas(4) uint32_t lagrangianPointCount = 0;
    alignas(4) float couplingStrength = 0.1f;
    alignas(4) uint32_t rigidBodyCount = 0;
    alignas(4) uint32_t useEmitter = 0;
    alignas(4) float spawnRate = 10.0f;
    alignas(16) glm::vec4 emitterPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    alignas(16) glm::vec4 emitterVel = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
};

struct RigidBodyState {
    alignas(16) glm::vec4 position;
    alignas(16) glm::vec4 orientation;
    alignas(16) glm::vec4 lin_vel;
    alignas(16) glm::vec4 ang_vel;
    alignas(16) glm::mat4 modelMatrix;
};

struct RigidBodyInfo {
    alignas(4) float mass;
    alignas(4) float inv_mass;
    alignas(16) glm::mat4 invInertiaTensor;
    alignas(4) float volume;
    alignas(4) uint32_t manualMode;
    alignas(16) glm::vec4 manualLinVel;
    alignas(16) glm::vec4 manualAngVel;
};

struct RenderingUBO {
    uint32_t Nx = 0;
    uint32_t Ny = 0;
    uint32_t Nz = 0;
    uint32_t render_mode = 0;
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct Particle {
    alignas(16) glm::vec4 position;  // xyz: position, w: age (current lifetime in seconds)
    alignas(16) glm::vec4 color;     // rgb: color, a: life (total lifetime in seconds)

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);

        return attributeDescriptions;
    }
};

struct Vertex {
    alignas(16) glm::vec4 pos;
    alignas(16) glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct MeshVertex {
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 normal;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(MeshVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(MeshVertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(MeshVertex, normal);

        return attributeDescriptions;
    }
};

struct LagrangianPoint {
    alignas(16) glm::vec4 position;
};

struct LagrangianData {
    alignas(16) glm::vec4 velocity;
    alignas(16) glm::vec4 force;
};

struct TotalForceTorque {
    alignas(16) glm::vec4 total_force;
    alignas(16) glm::vec4 total_torque;
};

enum class RigidBodyShape {
    SPHERE,
    BOX,
    CYLINDER,
    MESH
};

void generateSphere(float R, float target_spacing,
                    std::vector<glm::vec3>& positions, std::vector<float>& sk_buffer) {
    float total_area = 4.0f * PI * R * R;
    int N = (std::max)(10, (int)std::round(total_area / (target_spacing * target_spacing)));
    
    float sk = total_area / N;
    
    const float goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float angleIncrement = 2.0f * PI * goldenRatio;

    for (int i = 0; i < N; ++i) {
        float y = 1.0f - (i * 2.0f) / (N - 1);
        float radiusAtY = std::sqrt(1.0f - y * y);
        float theta = angleIncrement * i;

        float x = std::cos(theta) * radiusAtY;
        float z = std::sin(theta) * radiusAtY;

        positions.push_back(glm::vec3(x * R, y * R, z * R));
        sk_buffer.push_back(sk);
    }
}

void generateBox(glm::vec3 size, float target_spacing,
                 std::vector<glm::vec3>& positions, std::vector<float>& sk_buffer) {
    auto generateFace = [&](glm::vec3 center, glm::vec3 right, glm::vec3 up, float w, float h) {
        int nx = (std::max)(1, (int)std::round(w / target_spacing));
        int ny = (std::max)(1, (int)std::round(h / target_spacing));
        
        float sk = (w * h) / (nx * ny);
        float dx = w / nx;
        float dy = h / ny;
        
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                float u = -w/2.0f + (i + 0.5f) * dx;
                float v = -h/2.0f + (j + 0.5f) * dy;
                positions.push_back(center + right * u + up * v);
                sk_buffer.push_back(sk);
            }
        }
    };

    float W = size.x, H = size.y, D = size.z;
    generateFace(glm::vec3(0, 0,  D/2), glm::vec3(1,0,0), glm::vec3(0,1,0), W, H);
    generateFace(glm::vec3(0, 0, -D/2), glm::vec3(1,0,0), glm::vec3(0,1,0), W, H);
    generateFace(glm::vec3(0,  H/2, 0), glm::vec3(1,0,0), glm::vec3(0,0,1), W, D);
    generateFace(glm::vec3(0, -H/2, 0), glm::vec3(1,0,0), glm::vec3(0,0,1), W, D);
    generateFace(glm::vec3(  W/2, 0, 0), glm::vec3(0,1,0), glm::vec3(0,0,1), H, D);
    generateFace(glm::vec3( -W/2, 0, 0), glm::vec3(0,1,0), glm::vec3(0,0,1), H, D);
}

void generateCylinder(float R, float H, float target_spacing,
                      std::vector<glm::vec3>& positions, std::vector<float>& sk_buffer) {
    float side_area = 2.0f * PI * R * H;
    int nz = (std::max)(1, (int)std::round(H / target_spacing));
    int n_circ = (std::max)(6, (int)std::round((2.0f * PI * R) / target_spacing));
    float sk_side = side_area / (nz * n_circ);
    
    for (int i = 0; i < nz; ++i) {
        float z = -H/2.0f + (i + 0.5f) * (H / nz);
        for (int j = 0; j < n_circ; ++j) {
            float theta = j * (2.0f * PI / n_circ);
            positions.push_back(glm::vec3(R * std::cos(theta), R * std::sin(theta), z));
            sk_buffer.push_back(sk_side);
        }
    }

    auto generateCap = [&](float z_offset) {
        float cap_area = PI * R * R;
        int n_cap = (std::max)(1, (int)std::round(cap_area / (target_spacing * target_spacing)));
        float sk_cap = cap_area / n_cap;
        
        const float goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
        
        for (int i = 0; i < n_cap; ++i) {
            float r = R * std::sqrt((i + 0.5f) / n_cap);
            float theta = 2.0f * PI * goldenRatio * i;
            
            positions.push_back(glm::vec3(r * std::cos(theta), r * std::sin(theta), z_offset));
            sk_buffer.push_back(sk_cap);
        }
    };

    generateCap( H/2.0f);
    generateCap(-H/2.0f);
}

struct MeshParticleData {
    glm::vec3 position;
    glm::vec3 normal;
    float sk;
};

bool loadMeshFromCSV(const std::string& filepath,
                     std::vector<glm::vec3>& positions,
                     std::vector<float>& sk_buffer,
                     std::vector<glm::vec3>& normals) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        MeshParticleData data;

        std::getline(ss, token, ',');
        data.position.x = std::stof(token);
        std::getline(ss, token, ',');
        data.position.y = std::stof(token);
        std::getline(ss, token, ',');
        data.position.z = std::stof(token);
        std::getline(ss, token, ',');
        data.normal.x = std::stof(token);
        std::getline(ss, token, ',');
        data.normal.y = std::stof(token);
        std::getline(ss, token, ',');
        data.normal.z = std::stof(token);
        std::getline(ss, token, ',');
        data.sk = std::stof(token);

        positions.push_back(data.position);
        normals.push_back(data.normal);
        sk_buffer.push_back(data.sk);
    }

    std::cout << "Loaded " << positions.size() << " particles from CSV: " << filepath << std::endl;
    return true;
}

bool loadMeshFromBIN(const std::string& filepath,
                     std::vector<glm::vec3>& positions,
                     std::vector<float>& sk_buffer,
                     std::vector<glm::vec3>& normals) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open BIN file: " << filepath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    const size_t floatsPerParticle = 7;
    size_t numParticles = fileSize / (floatsPerParticle * sizeof(float));

    if (fileSize % (floatsPerParticle * sizeof(float)) != 0) {
        std::cerr << "Warning: BIN file size not aligned with particle data size" << std::endl;
    }

    std::vector<float> buffer(numParticles * floatsPerParticle);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    positions.resize(numParticles);
    normals.resize(numParticles);
    sk_buffer.resize(numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        size_t offset = i * floatsPerParticle;
        positions[i] = glm::vec3(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2]);
        normals[i] = glm::vec3(buffer[offset + 3], buffer[offset + 4], buffer[offset + 5]);
        sk_buffer[i] = buffer[offset + 6];
    }

    std::cout << "Loaded " << numParticles << " particles from BIN: " << filepath << std::endl;
    return true;
}

bool loadMeshFromGLB(const std::string& filepath,
                     std::vector<glm::vec3>& positions,
                     std::vector<float>& sk_buffer,
                     std::vector<glm::vec3>& normals) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open GLB file: " << filepath << std::endl;
        return false;
    }

    uint32_t magic, version, totalLength;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&totalLength), 4);

    if (magic != 0x46546C67) {
        std::cerr << "Invalid GLB file: wrong magic number" << std::endl;
        return false;
    }

    uint32_t jsonChunkLength, jsonChunkType;
    file.read(reinterpret_cast<char*>(&jsonChunkLength), 4);
    file.read(reinterpret_cast<char*>(&jsonChunkType), 4);

    if (jsonChunkType != 0x4E4F534A) {
        std::cerr << "Invalid GLB file: expected JSON chunk" << std::endl;
        return false;
    }

    std::string jsonContent(jsonChunkLength, '\0');
    file.read(&jsonContent[0], jsonChunkLength);

    uint32_t binChunkLength = 0, binChunkType = 0;
    file.read(reinterpret_cast<char*>(&binChunkLength), 4);
    file.read(reinterpret_cast<char*>(&binChunkType), 4);

    if (binChunkType != 0x004E4942) {
        std::cerr << "Invalid GLB file: expected BIN chunk" << std::endl;
        return false;
    }

    auto extractNumber = [](const std::string& json, const std::string& key, size_t startPos = 0) -> int {
        std::string searchStr = "\"" + key + "\"";
        size_t pos = json.find(searchStr, startPos);
        if (pos == std::string::npos) return -1;
        
        size_t colonPos = json.find(':', pos);
        if (colonPos == std::string::npos) return -1;
        
        size_t valueStart = json.find_first_of("-0123456789", colonPos);
        if (valueStart == std::string::npos) return -1;
        
        size_t valueEnd = json.find_first_not_of("-0123456789", valueStart);
        if (valueEnd == std::string::npos) valueEnd = json.length();
        
        try {
            return std::stoi(json.substr(valueStart, valueEnd - valueStart));
        } catch (...) {
            return -1;
        }
    };

    int vertexCount = -1;
    size_t accessorPos = jsonContent.find("\"accessors\"");
    if (accessorPos != std::string::npos) {
        size_t countPos = jsonContent.find("\"count\"", accessorPos);
        if (countPos != std::string::npos) {
            vertexCount = extractNumber(jsonContent, "count", countPos);
        }
    }
    
    int bufferViewByteOffset = 0;
    size_t bufferViewPos = jsonContent.find("\"bufferViews\"");
    if (bufferViewPos != std::string::npos) {
        size_t byteOffsetPos = jsonContent.find("\"byteOffset\"", bufferViewPos);
        if (byteOffsetPos != std::string::npos) {
            bufferViewByteOffset = extractNumber(jsonContent, "byteOffset", byteOffsetPos);
        }
    }
    
    int accessorByteOffset = 0;
    if (accessorPos != std::string::npos) {
        size_t byteOffsetPos = jsonContent.find("\"byteOffset\"", accessorPos);
        if (byteOffsetPos != std::string::npos && byteOffsetPos < jsonContent.find("]", accessorPos)) {
            accessorByteOffset = extractNumber(jsonContent, "byteOffset", byteOffsetPos);
        }
    }

    if (vertexCount <= 0) {
        vertexCount = binChunkLength / sizeof(glm::vec3);
    }
    
    size_t totalVertices = static_cast<size_t>(vertexCount);
    size_t particleCount = totalVertices / 2;
    
    if (particleCount == 0) {
        std::cerr << "Invalid GLB: no particle data found" << std::endl;
        return false;
    }

    size_t binDataOffset = 12 + 8 + jsonChunkLength + 8;
    size_t dataOffset = binDataOffset + bufferViewByteOffset + accessorByteOffset;
    
    std::cout << "GLB Debug: vertexCount=" << vertexCount 
              << ", bufferViewByteOffset=" << bufferViewByteOffset
              << ", accessorByteOffset=" << accessorByteOffset
              << ", binDataOffset=" << binDataOffset
              << ", dataOffset=" << dataOffset << std::endl;
    
    file.seekg(dataOffset, std::ios::beg);
    
    std::vector<glm::vec3> allVertices(totalVertices);
    file.read(reinterpret_cast<char*>(allVertices.data()), totalVertices * sizeof(glm::vec3));

    positions.resize(particleCount);
    sk_buffer.resize(particleCount);
    normals.resize(particleCount, glm::vec3(0.0f));

    for (size_t i = 0; i < particleCount; ++i) {
        positions[i] = allVertices[i];
        sk_buffer[i] = allVertices[i + particleCount].x;
    }

    std::cout << "Loaded " << particleCount << " particles from GLB: " << filepath << std::endl;
    std::cout << "Total vertices in GLB: " << totalVertices << std::endl;
    
    float skSum = 0.0f;
    for (size_t i = 0; i < std::min(size_t(5), particleCount); ++i) {
        std::cout << "  sk[" << i << "] = " << sk_buffer[i] << std::endl;
        skSum += sk_buffer[i];
    }
    float totalSk = 0.0f;
    for (float sk : sk_buffer) totalSk += sk;
    std::cout << "Total sk sum: " << totalSk << std::endl;
    
    return true;
}

bool loadMeshTrianglesFromGLB(const std::string& filepath,
                              std::vector<MeshVertex>& vertices,
                              std::vector<uint32_t>& indices) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open GLB file: " << filepath << std::endl;
        return false;
    }

    uint32_t magic, version, totalLength;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&totalLength), 4);

    if (magic != 0x46546C67) {
        std::cerr << "Invalid GLB file: wrong magic number" << std::endl;
        return false;
    }

    uint32_t jsonChunkLength, jsonChunkType;
    file.read(reinterpret_cast<char*>(&jsonChunkLength), 4);
    file.read(reinterpret_cast<char*>(&jsonChunkType), 4);

    if (jsonChunkType != 0x4E4F534A) {
        std::cerr << "Invalid GLB file: expected JSON chunk" << std::endl;
        return false;
    }

    std::string jsonContent(jsonChunkLength, '\0');
    file.read(&jsonContent[0], jsonChunkLength);

    uint32_t binChunkLength = 0, binChunkType = 0;
    file.read(reinterpret_cast<char*>(&binChunkLength), 4);
    file.read(reinterpret_cast<char*>(&binChunkType), 4);

    if (binChunkType != 0x004E4942) {
        std::cerr << "Invalid GLB file: expected BIN chunk" << std::endl;
        return false;
    }

    auto extractNumber = [](const std::string& json, const std::string& key, size_t startPos = 0) -> int {
        std::string searchStr = "\"" + key + "\"";
        size_t pos = json.find(searchStr, startPos);
        if (pos == std::string::npos) return -1;
        
        size_t colonPos = json.find(':', pos);
        if (colonPos == std::string::npos) return -1;
        
        size_t valueStart = json.find_first_of("-0123456789", colonPos);
        if (valueStart == std::string::npos) return -1;
        
        size_t valueEnd = json.find_first_not_of("-0123456789", valueStart);
        if (valueEnd == std::string::npos) valueEnd = json.length();
        
        try {
            return std::stoi(json.substr(valueStart, valueEnd - valueStart));
        } catch (...) {
            return -1;
        }
    };

    auto findAccessorIndex = [](const std::string& json, const std::string& semantic) -> int {
        size_t primitivesPos = json.find("\"primitives\"");
        if (primitivesPos == std::string::npos) return -1;
        
        size_t attributesPos = json.find("\"attributes\"", primitivesPos);
        if (attributesPos == std::string::npos) return -1;
        
        std::string searchStr = "\"" + semantic + "\"";
        size_t semanticPos = json.find(searchStr, attributesPos);
        if (semanticPos == std::string::npos) return -1;
        
        size_t colonPos = json.find(':', semanticPos);
        if (colonPos == std::string::npos) return -1;
        
        size_t valueStart = json.find_first_of("-0123456789", colonPos);
        if (valueStart == std::string::npos) return -1;
        
        size_t valueEnd = json.find_first_not_of("-0123456789", valueStart);
        if (valueEnd == std::string::npos) valueEnd = json.length();
        
        try {
            return std::stoi(json.substr(valueStart, valueEnd - valueStart));
        } catch (...) {
            return -1;
        }
    };

    int positionAccessorIdx = findAccessorIndex(jsonContent, "POSITION");
    int normalAccessorIdx = findAccessorIndex(jsonContent, "NORMAL");
    
    size_t primitivesPos = jsonContent.find("\"primitives\"");
    int indicesAccessorIdx = -1;
    if (primitivesPos != std::string::npos) {
        size_t indicesPos = jsonContent.find("\"indices\"", primitivesPos);
        if (indicesPos != std::string::npos) {
            size_t colonPos = jsonContent.find(':', indicesPos);
            if (colonPos != std::string::npos) {
                size_t valueStart = jsonContent.find_first_of("-0123456789", colonPos);
                if (valueStart != std::string::npos) {
                    size_t valueEnd = jsonContent.find_first_not_of("-0123456789", valueStart);
                    if (valueEnd == std::string::npos) valueEnd = jsonContent.length();
                    try {
                        indicesAccessorIdx = std::stoi(jsonContent.substr(valueStart, valueEnd - valueStart));
                    } catch (...) {}
                }
            }
        }
    }
    
    std::cout << "GLB Mesh Debug: positionAccessor=" << positionAccessorIdx 
              << ", normalAccessor=" << normalAccessorIdx
              << ", indicesAccessor=" << indicesAccessorIdx << std::endl;

    auto getAccessorInfo = [&](int accessorIdx) -> std::tuple<int, int, int, int> {
        if (accessorIdx < 0) return {-1, -1, -1, -1};
        
        std::string accessorSearch = "\"accessors\"";
        size_t accessorPos = jsonContent.find(accessorSearch);
        if (accessorPos == std::string::npos) return {-1, -1, -1, -1};
        
        size_t startBracket = jsonContent.find('[', accessorPos);
        size_t currentBracket = startBracket;
        int currentIdx = 0;
        
        while (currentIdx <= accessorIdx) {
            size_t nextBracket = jsonContent.find('{', currentBracket);
            if (nextBracket == std::string::npos) break;
            
            if (currentIdx == accessorIdx) {
                size_t endBracket = jsonContent.find('}', nextBracket);
                std::string accessorStr = jsonContent.substr(nextBracket, endBracket - nextBracket);
                
                int count = -1, bufferView = -1, byteOffset = 0, componentType = -1;
                
                size_t countPos = accessorStr.find("\"count\"");
                if (countPos != std::string::npos) {
                    count = extractNumber(accessorStr, "count", countPos);
                }
                
                size_t bvPos = accessorStr.find("\"bufferView\"");
                if (bvPos != std::string::npos) {
                    bufferView = extractNumber(accessorStr, "bufferView", bvPos);
                }
                
                size_t boPos = accessorStr.find("\"byteOffset\"");
                if (boPos != std::string::npos) {
                    byteOffset = extractNumber(accessorStr, "byteOffset", boPos);
                }
                
                size_t ctPos = accessorStr.find("\"componentType\"");
                if (ctPos != std::string::npos) {
                    componentType = extractNumber(accessorStr, "componentType", ctPos);
                }
                
                return {count, bufferView, byteOffset, componentType};
            }
            
            currentBracket = jsonContent.find(',', nextBracket);
            if (currentBracket == std::string::npos) break;
            currentIdx++;
        }
        
        return {-1, -1, -1, -1};
    };

    auto getBufferViewInfo = [&](int bufferViewIdx) -> std::tuple<int, int> {
        if (bufferViewIdx < 0) return {-1, -1};
        
        std::string bvSearch = "\"bufferViews\"";
        size_t bvPos = jsonContent.find(bvSearch);
        if (bvPos == std::string::npos) return {-1, -1};
        
        size_t startBracket = jsonContent.find('[', bvPos);
        size_t currentBracket = startBracket;
        int currentIdx = 0;
        
        while (currentIdx <= bufferViewIdx) {
            size_t nextBracket = jsonContent.find('{', currentBracket);
            if (nextBracket == std::string::npos) break;
            
            if (currentIdx == bufferViewIdx) {
                size_t endBracket = jsonContent.find('}', nextBracket);
                std::string bvStr = jsonContent.substr(nextBracket, endBracket - nextBracket);
                
                int byteOffset = 0, byteLength = -1;
                
                size_t offsetPos = bvStr.find("\"byteOffset\"");
                if (offsetPos != std::string::npos) {
                    byteOffset = extractNumber(bvStr, "byteOffset", offsetPos);
                }
                
                size_t lengthPos = bvStr.find("\"byteLength\"");
                if (lengthPos != std::string::npos) {
                    byteLength = extractNumber(bvStr, "byteLength", lengthPos);
                }
                
                return {byteOffset, byteLength};
            }
            
            currentBracket = jsonContent.find(',', nextBracket);
            if (currentBracket == std::string::npos) break;
            currentIdx++;
        }
        
        return {-1, -1};
    };

    size_t binDataOffset = 12 + 8 + jsonChunkLength + 8;

    auto [posCount, posBufferView, posByteOffset, posComponentType] = getAccessorInfo(positionAccessorIdx);
    auto [normCount, normBufferView, normByteOffset, normComponentType] = getAccessorInfo(normalAccessorIdx);
    auto [idxCount, idxBufferView, idxByteOffset, idxComponentType] = getAccessorInfo(indicesAccessorIdx);

    if (posCount <= 0) {
        std::cerr << "Failed to find position data in GLB" << std::endl;
        return false;
    }

    auto [posBvOffset, posBvLength] = getBufferViewInfo(posBufferView);
    auto [normBvOffset, normBvLength] = getBufferViewInfo(normBufferView);
    auto [idxBvOffset, idxBvLength] = getBufferViewInfo(idxBufferView);

    vertices.resize(posCount);
    
    size_t posDataOffset = binDataOffset + posBvOffset + posByteOffset;
    file.seekg(posDataOffset, std::ios::beg);
    std::vector<glm::vec3> positions(posCount);
    file.read(reinterpret_cast<char*>(positions.data()), posCount * sizeof(glm::vec3));
    
    for (int i = 0; i < posCount; ++i) {
        vertices[i].pos = positions[i];
    }

    if (normCount > 0 && normBufferView >= 0) {
        size_t normDataOffset = binDataOffset + normBvOffset + normByteOffset;
        file.seekg(normDataOffset, std::ios::beg);
        std::vector<glm::vec3> normals(normCount);
        file.read(reinterpret_cast<char*>(normals.data()), normCount * sizeof(glm::vec3));
        
        for (int i = 0; i < std::min(posCount, normCount); ++i) {
            vertices[i].normal = normals[i];
        }
    } else {
        for (int i = 0; i < posCount; ++i) {
            vertices[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }
    }

    if (idxCount > 0 && idxBufferView >= 0) {
        size_t idxDataOffset = binDataOffset + idxBvOffset + idxByteOffset;
        file.seekg(idxDataOffset, std::ios::beg);
        
        if (idxComponentType == 5123) {
            std::vector<uint16_t> indices16(idxCount);
            file.read(reinterpret_cast<char*>(indices16.data()), idxCount * sizeof(uint16_t));
            indices.resize(idxCount);
            for (int i = 0; i < idxCount; ++i) {
                indices[i] = static_cast<uint32_t>(indices16[i]);
            }
            std::cout << "  Index type: uint16" << std::endl;
        } else if (idxComponentType == 5125) {
            indices.resize(idxCount);
            file.read(reinterpret_cast<char*>(indices.data()), idxCount * sizeof(uint32_t));
            std::cout << "  Index type: uint32" << std::endl;
        } else if (idxComponentType == 5121) {
            std::vector<uint8_t> indices8(idxCount);
            file.read(reinterpret_cast<char*>(indices8.data()), idxCount * sizeof(uint8_t));
            indices.resize(idxCount);
            for (int i = 0; i < idxCount; ++i) {
                indices[i] = static_cast<uint32_t>(indices8[i]);
            }
            std::cout << "  Index type: uint8" << std::endl;
        } else {
            std::cerr << "Unknown index component type: " << idxComponentType << std::endl;
            indices.clear();
            for (int i = 0; i < posCount; ++i) {
                indices.push_back(i);
            }
        }
    } else {
        indices.clear();
        for (int i = 0; i < posCount; ++i) {
            indices.push_back(i);
        }
    }

    std::cout << "Loaded mesh from GLB: " << filepath << std::endl;
    std::cout << "  Vertices: " << posCount << std::endl;
    std::cout << "  Indices: " << indices.size() << std::endl;
    std::cout << "  Triangles: " << indices.size() / 3 << std::endl;

    return true;
}

struct RigidBody {
    RigidBodyShape shape = RigidBodyShape::SPHERE;
    float rho;
    float radius;
    glm::vec3 boxSize;
    float cylinderRadius;
    float cylinderHeight;
    float volume;
    float mass;
    float inv_mass;
    glm::mat3 inertiaTensor;
    glm::mat3 invInertiaTensor;
    glm::vec3 position;
    glm::quat orientation;
    glm::vec3 linear_velocity;
    glm::vec3 angular_velocity;
    std::string meshFilePath;
    glm::vec3 meshBoundingBoxMin;
    glm::vec3 meshBoundingBoxMax;
    float meshScale;
    int meshCoordSystem;
    bool isManualControl = true;
    float manualLinVel[3] = {0.0f, 0.0f, 0.0f};
    float manualAngVel[3] = {0.0f, 0.0f, 0.0f};

    RigidBody() {
        shape = RigidBodyShape::MESH;
        rho = 1.0f;
        radius = 10.0f;
        boxSize = glm::vec3(40.0f, 3.0f, 25.0f);
        cylinderRadius = 10.0f;
        cylinderHeight = 20.0f;
        meshFilePath = "models/fan_pointcloud.glb";
        meshBoundingBoxMin = glm::vec3(-1.0f);
        meshBoundingBoxMax = glm::vec3(1.0f);
        meshScale = 40.0f;
        meshCoordSystem = 1;
        isManualControl = true;
        updateInertia();
        float travel_distance = 48.0f;
        float a_point = Ny / 2.0f - travel_distance / 2.0f;
        position = glm::vec3(Nx / 2.0f, Ny / 2.0f, Nz / 2.0f);
        orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        linear_velocity = glm::vec3(0.0f);
        angular_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    void updateInertia() {
        switch (shape) {
            case RigidBodyShape::SPHERE:
                volume = 4.0f / 3.0f * PI * radius * radius * radius;
                mass = rho * volume;
                inv_mass = 1.0f / mass;
                {
                    float I = 0.4f * mass * radius * radius;
                    inertiaTensor = glm::mat3(I);
                    invInertiaTensor = glm::mat3(1.0f / I);
                }
                break;
            case RigidBodyShape::BOX:
                volume = boxSize.x * boxSize.y * boxSize.z;
                mass = rho * volume;
                inv_mass = 1.0f / mass;
                {
                    float Ix = mass * (boxSize.y * boxSize.y + boxSize.z * boxSize.z) / 12.0f;
                    float Iy = mass * (boxSize.x * boxSize.x + boxSize.z * boxSize.z) / 12.0f;
                    float Iz = mass * (boxSize.x * boxSize.x + boxSize.y * boxSize.y) / 12.0f;
                    inertiaTensor = glm::mat3(Ix, 0, 0, 0, Iy, 0, 0, 0, Iz);
                    invInertiaTensor = glm::mat3(1.0f/Ix, 0, 0, 0, 1.0f/Iy, 0, 0, 0, 1.0f/Iz);
                }
                break;
            case RigidBodyShape::CYLINDER:
                volume = PI * cylinderRadius * cylinderRadius * cylinderHeight;
                mass = rho * volume;
                inv_mass = 1.0f / mass;
                {
                    float I_axis = 0.5f * mass * cylinderRadius * cylinderRadius;
                    float I_perp = mass * (3.0f * cylinderRadius * cylinderRadius + cylinderHeight * cylinderHeight) / 12.0f;
                    inertiaTensor = glm::mat3(I_perp, 0, 0, 0, I_perp, 0, 0, 0, I_axis);
                    invInertiaTensor = glm::mat3(1.0f/I_perp, 0, 0, 0, 1.0f/I_perp, 0, 0, 0, 1.0f/I_axis);
                }
                break;
            case RigidBodyShape::MESH:
                volume = (meshBoundingBoxMax.x - meshBoundingBoxMin.x) *
                         (meshBoundingBoxMax.y - meshBoundingBoxMin.y) *
                         (meshBoundingBoxMax.z - meshBoundingBoxMin.z);
                mass = rho * volume;
                inv_mass = 1.0f / mass;
                {
                    float W = meshBoundingBoxMax.x - meshBoundingBoxMin.x;
                    float H = meshBoundingBoxMax.y - meshBoundingBoxMin.y;
                    float D = meshBoundingBoxMax.z - meshBoundingBoxMin.z;
                    float Ix = mass * (H * H + D * D) / 12.0f;
                    float Iy = mass * (W * W + D * D) / 12.0f;
                    float Iz = mass * (W * W + H * H) / 12.0f;
                    inertiaTensor = glm::mat3(Ix, 0, 0, 0, Iy, 0, 0, 0, Iz);
                    invInertiaTensor = glm::mat3(1.0f/Ix, 0, 0, 0, 1.0f/Iy, 0, 0, 0, 1.0f/Iz);
                }
                break;
        }
    }

    RigidBodyState getRigidBodyState() const {
        RigidBodyState state;
        state.position = glm::vec4(position, 1.0f);
        state.orientation = glm::vec4(orientation.x, orientation.y, orientation.z, orientation.w);
        state.lin_vel = glm::vec4(linear_velocity, 0.0f);
        state.ang_vel = glm::vec4(angular_velocity, 0.0f);
        glm::mat4 M = glm::mat4(1.0f);
        M = glm::translate(M, position);
        M = M * glm::mat4_cast(orientation);
        state.modelMatrix = M;
        return state;
    }

    RigidBodyInfo getRigidBodyInfo() const {
        RigidBodyInfo info;
        info.mass = mass;
        info.inv_mass = inv_mass;
        info.invInertiaTensor = glm::mat4(invInertiaTensor);
        info.volume = volume;
        info.manualMode = isManualControl ? 1u : 0u;
        info.manualLinVel = glm::vec4(manualLinVel[0], manualLinVel[1], manualLinVel[2], 0.0f);
        info.manualAngVel = glm::vec4(manualAngVel[0], manualAngVel[1], manualAngVel[2], 0.0f);
        return info;
    }
};

class ComputeShaderApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    bool firstMouse = true;
    bool mouseFree = false;
    double lastX = 0, lastY = 0;
    float rx = 0, ry = 90;
    float distance = 2.0f;
    bool framebufferResized = false;
    bool isInit = false;
    uint32_t currentTime = 0;
    int render_mode = 2;
    int model_render_mode = 1; // 0: Particles, 1: Mesh
    bool enIBM = true;
    float couplingStrength = 1.0f;
    bool isManualControl = true;
    float manualLinVel[3] = { 0.0f, 0.0f, 0.0f };
    float manualAngVel[3] = { 0.0f, 0.0f, 0.0f };
    float rigidBodyDensity = 1.0f;
    uint32_t useEmitter = 1;
    float spawnRate = 200.0f;
    float emitterPos[4] = { Nx / 2.0f, 30.0f, Nz / 2.0f, 1.0f };
    float emitterVel[4] = { 0.0f, 0.0f, 0.15f, 0.0f };

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage skyboxImage;
    VkDeviceMemory skyboxImageMemory;
    VkImageView skyboxImageView;
    VkSampler skyboxSampler;

    VkRenderPass renderPass;

    VkDescriptorSetLayout graphicsDescriptorSetLayout;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout graphicsPipelineLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline graphicsPipeline;
    VkPipeline diagnosticPipeline;
    VkPipeline wireframePipeline;
    VkPipeline skyboxPipeline;
    VkPipeline lagrangianPipeline;
    VkPipeline velocityPipeline;
    VkPipeline computePipeline;
    VkPipeline initPipeline;
    VkPipeline collideAndStreamPipeline;
    VkPipeline ibmForce1Pipeline;
    VkPipeline ibmForce2Pipeline;
    VkPipeline calcUPipeline;
    VkPipeline updatePositionsPipeline;
    VkPipeline forceReductionPipeline;
    VkPipeline rigidBodySolverPipeline;

    VkCommandPool commandPool;

    std::vector<VkBuffer> shaderStorageBuffers;
    std::vector<VkDeviceMemory> shaderStorageBuffersMemory;
    std::vector<VkBuffer> velocityBuffers;
    std::vector<VkDeviceMemory> velocityBuffersMemory;
    std::vector<VkBuffer> tempVelBuffers;
    std::vector<VkDeviceMemory> tempVelBuffersMemory;
    std::vector<VkBuffer> rhoBuffers;
    std::vector<VkDeviceMemory> rhoBuffersMemory;
    std::vector<VkBuffer> flagBuffers;
    std::vector<VkDeviceMemory> flagBuffersMemory;
    std::vector<VkBuffer> DDFBuffers;
    std::vector<VkDeviceMemory> DDFBuffersMemory;
    std::vector<VkBuffer> borderForceBuffers;
    std::vector<VkDeviceMemory> borderForceBuffersMemory;

    std::vector<VkBuffer> lagrangianPointsBuffers;
    std::vector<VkDeviceMemory> lagrangianPointsBuffersMemory;
    std::vector<VkBuffer> lagrangianPointsRestBuffers;
    std::vector<VkDeviceMemory> lagrangianPointsRestBuffersMemory;
    std::vector<VkBuffer> lagrangianPointsPrevBuffers;
    std::vector<VkDeviceMemory> lagrangianPointsPrevBuffersMemory;
    std::vector<VkBuffer> lagrangianDataBuffers;
    std::vector<VkDeviceMemory> lagrangianDataBuffersMemory;
    std::vector<VkBuffer> tempForcesBuffers;
    std::vector<VkDeviceMemory> tempForcesBuffersMemory;

    std::vector<VkBuffer> skBuffers;
    std::vector<VkDeviceMemory> skBuffersMemory;

    std::vector<VkBuffer> totalForceTorqueBuffers;
    std::vector<VkDeviceMemory> totalForceTorqueBuffersMemory;
    std::vector<void*> totalForceTorqueBuffersMapped;

    std::vector<VkBuffer> rigidBodyStateBuffers;
    std::vector<VkDeviceMemory> rigidBodyStateBuffersMemory;
    std::vector<VkBuffer> rigidBodyInfoBuffers;
    std::vector<VkDeviceMemory> rigidBodyInfoBuffersMemory;
    std::vector<VkBuffer> bodyIndexBuffers;
    std::vector<VkDeviceMemory> bodyIndexBuffersMemory;

    std::vector<RigidBody> rigidBodies;
    uint32_t rigidBodyCount = 0;

    std::vector<VkBuffer> wireframeBuffers;
    std::vector<VkDeviceMemory> wireframeBuffersMemory;
    std::vector<VkBuffer> wireframeIndexBuffers;
    std::vector<VkDeviceMemory> wireframeIndexBuffersMemory;

    std::vector<std::vector<MeshVertex>> meshVerticesPerBody;
    std::vector<std::vector<uint32_t>> meshIndicesPerBody;
    std::vector<VkBuffer> meshVertexBuffers;
    std::vector<VkDeviceMemory> meshVertexBuffersMemory;
    std::vector<VkBuffer> meshIndexBuffers;
    std::vector<VkDeviceMemory> meshIndexBuffersMemory;
    VkPipeline meshPipeline;

    std::vector<VkBuffer> skyboxBuffers;
    std::vector<VkDeviceMemory> skyboxBuffersMemory;
    std::vector<VkBuffer> skyboxIndexBuffers;
    std::vector<VkDeviceMemory> skyboxIndexBuffersMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::vector<VkBuffer> renderingUBOBuffers;
    std::vector<VkDeviceMemory> renderingUBOBuffersMemory;
    std::vector<void*> renderingUBOBuffersMapped;

    VkDescriptorPool descriptorPool;
    VkDescriptorPool imguiDescriptorPool;
    std::vector<VkDescriptorSet> graphicsDescriptorSets;
    std::vector<VkDescriptorSet> computeDescriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> computeInFlightFences;
    uint32_t currentFrame = 0;

    float lastFrameTime = 0.0f;
    float totalTime = 0.0f;
    double lastTime = 0.0f;

    uint32_t particle_count = 0;

    std::vector<Vertex> wireframeVertices = {
        {{  0,  0,  0,  0 }, { 1,  1,  1,  1 }},  // 0
        {{ Nx,  0,  0,  0 }, { 1,  1,  1,  1 }},  // 1
        {{  0, Ny,  0,  0 }, { 1,  1,  1,  1 }},  // 2
        {{  0,  0, Nz,  0 }, { 1,  1,  1,  1 }},  // 3
        {{ Nx, Ny,  0,  0 }, { 1,  1,  1,  1 }},  // 4
        {{ Nx,  0, Nz,  0 }, { 1,  1,  1,  1 }},  // 5
        {{  0, Ny, Nz,  0 }, { 1,  1,  1,  1 }},  // 6
        {{ Nx, Ny, Nz,  0 }, { 1,  1,  1,  1 }},  // 7
    };

    std::vector<uint32_t> wireframeIndices = {
         0,  1,
         1,  4,
         4,  2,
         2,  0,
         1,  5,
         0,  3,
         4,  7,
         2,  6,
         3,  5,
         5,  7,
         6,  7,
         3,  6,
    };

    std::vector<glm::vec3> skyboxVertices = {
        { -500,  500, -500,},
        { -500, -500, -500,},
        {  500, -500, -500,},
        {  500,  500, -500,},
        { -500,  500,  500,},
        {  500,  500,  500,},
        {  500, -500,  500,},
        { -500, -500,  500,},
        { -500,  500, -500,},
        {  500,  500, -500,},
        {  500,  500,  500,},
        { -500,  500,  500,},
        {  500,  500, -500,},
        {  500, -500, -500,},
        {  500, -500,  500,},
        {  500,  500,  500,},
        {  500, -500, -500,},
        { -500, -500, -500,},
        { -500, -500,  500,},
        {  500, -500,  500,},
        { -500, -500, -500,},
        { -500,  500, -500,},
        { -500,  500,  500,},
        { -500, -500,  500,},
    };

    std::vector<uint32_t> skyboxIndices = {
         0,
         1,
         2,
         2,
         3,
         0,
         4,
         5,
         6,
         6,
         7,
         4,
         8,
         9,
        10,
        10,
        11,
         8,
        12,
        13,
        14,
        14,
        15,
        12,
        16,
        17,
        18,
        18,
        19,
        16,
        20,
        21,
        22,
        22,
        23,
        20,
    };

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouse_move_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);

        lastTime = glfwGetTime();
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));
        WIDTH = width;
        HEIGHT = height;
        app->framebufferResized = true;
    }

    static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));

        if (!app->mouseFree) return;

        if (app->firstMouse) {
            app->lastX = xpos;
            app->lastY = ypos;
            app->firstMouse = false;
        }

        float xoffset = xpos - app->lastX;
        float yoffset = app->lastY - ypos;
        app->lastX = xpos;
        app->lastY = ypos;

        float sensitivity = 0.5f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        app->rx -= xoffset;
        app->ry += yoffset;

        if (app->ry > 179.0f)
            app->ry = 179.0f;
        if (app->ry < 1.0f)
            app->ry = 1.0f;
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));

        if (!app->mouseFree) return;
        if (yoffset > 0) {
            app->distance += 0.1f;
        }
        else if (yoffset < 0) {
            app->distance -= 0.1f;
        }
        if (app->distance < 0.1f) {
            app->distance = 0.1f;
        }
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));

        if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
            app->mouseFree = 1 - app->mouseFree;
            if (!app->mouseFree) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                glfwSetCursorPos(window, WIDTH / 2, HEIGHT / 2);
                app->firstMouse = true;
            }
            else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
        }
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsDescriptorSetLayout();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createSkyBoxPipeline();
        createComputePipeline();
        createLagrangianPipeline();
        createVelocityPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createSkybox();
        createVertexBuffers();
        createIndexBuffers();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createGraphicsDescriptorSets();
        createComputeDescriptorSets();
        createCommandBuffers();
        createComputeCommandBuffers();
        createSyncObjects();
        initImGui();
    }

    void initImGui() {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create imgui descriptor pool!");
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsAndComputeFamily.value();
        init_info.Queue = graphicsQueue;
        init_info.RenderPass = renderPass;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = imguiDescriptorPool;
        init_info.Subpass = 0;
        init_info.MinImageCount = 2;
        init_info.ImageCount = static_cast<uint32_t>(swapChainImages.size());
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;
        ImGui_ImplVulkan_Init(&init_info);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        ImGui_ImplVulkan_CreateFontsTexture();
        endSingleTimeCommands(commandBuffer);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            static double start_time = glfwGetTime();
            glfwPollEvents();
            
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            {
                ImGui::Begin("Particle Settings");
                const char* render_modes[] = { "Normal Particles", "Diagnostic Particles", "Velocity Field" };
                ImGui::Combo("Render Mode", &render_mode, render_modes, IM_ARRAYSIZE(render_modes));
                
                ImGui::Separator();
                ImGui::Text("Model Rendering");
                const char* model_render_modes[] = { "Particles", "Mesh" };
                ImGui::Combo("Model Render Mode", &model_render_mode, model_render_modes, IM_ARRAYSIZE(model_render_modes));
                
                ImGui::Separator();
                ImGui::Checkbox("Enable IBM?", &enIBM);
                ImGui::SliderFloat("Coupling Strength", &couplingStrength, 0.1f, 1.0f, "%.2f");
                ImGui::Separator();
                ImGui::Text("Emitter Settings");
                ImGui::Checkbox("Use Emitter Mode", (bool*)&useEmitter);
                if (useEmitter) {
                    ImGui::SliderFloat("Spawn Rate", &spawnRate, 100.0f, 5000.0f, "%.1f");
                    ImGui::SliderFloat4("Emitter Pos", emitterPos, 0.0f, 128.0f);
                }
                ImGui::Text("FPS: %.1f", 1000.0f / lastFrameTime);
                ImGui::Text("Particles: %d", particle_count);
                ImGui::End();
            }
            
            {
                ImGui::Begin("Rigid Body Control");
                
                if (rigidBodies.empty()) {
                    ImGui::Text("No rigid bodies in scene.");
                    ImGui::End();
                } else {
                    static int selectedBodyIndex = 0;
                    
                    ImGui::Text("Body Count: %u", rigidBodyCount);
                    
                    if (selectedBodyIndex >= (int)rigidBodies.size()) selectedBodyIndex = 0;
                    
                    for (size_t i = 0; i < rigidBodies.size(); i++) {
                        char label[32];
                        sprintf(label, "Body %zu", i);
                        if (ImGui::RadioButton(label, selectedBodyIndex == (int)i)) {
                            selectedBodyIndex = (int)i;
                        }
                        ImGui::SameLine();
                        ImGui::Text("Mass: %.2f", rigidBodies[i].mass);
                    }
                    
                    ImGui::Separator();
                    
                    RigidBody& body = rigidBodies[selectedBodyIndex];
                    
                    ImGui::Text("Selected: Body %d", selectedBodyIndex);
                    ImGui::Text("Position: (%.1f, %.1f, %.1f)", body.position.x, body.position.y, body.position.z);
                    
                    ImGui::Separator();
                    
                    if (ImGui::Checkbox("Manual Control", &body.isManualControl)) {
                        body.manualLinVel[0] = body.manualLinVel[1] = body.manualLinVel[2] = 0.0f;
                        body.manualAngVel[0] = body.manualAngVel[1] = body.manualAngVel[2] = 0.0f;
                    }
                    
                    ImGui::Separator();
                    
                    if (body.isManualControl) {
                        ImGui::Text("Manual Velocity Control");
                        ImGui::SliderFloat("vx", &body.manualLinVel[0], -0.05f, 0.05f);
                        ImGui::SliderFloat("vy", &body.manualLinVel[1], -0.05f, 0.05f);
                        ImGui::SliderFloat("vz", &body.manualLinVel[2], -0.05f, 0.05f);
                        
                        ImGui::Separator();
                        ImGui::Text("Manual Angular Velocity");
                        ImGui::SliderFloat("rva", &body.manualAngVel[0], -0.02f, 0.02f);
                        ImGui::SliderFloat("rvb", &body.manualAngVel[1], -0.02f, 0.02f);
                        ImGui::SliderFloat("rvc", &body.manualAngVel[2], -0.02f, 0.02f);
                        
                        if (ImGui::Button("Stop All")) {
                            body.manualLinVel[0] = body.manualLinVel[1] = body.manualLinVel[2] = 0.0f;
                            body.manualAngVel[0] = body.manualAngVel[1] = body.manualAngVel[2] = 0.0f;
                        }
                    } else {
                        ImGui::Text("Mode: Bi-directional Physics");
                    }
                    
                    ImGui::Separator();
                    
                    if (ImGui::SliderFloat("Density", &rigidBodyDensity, 0.1f, 2.0f, "%.2f")) {
                        for (auto& b : rigidBodies) {
                            b.rho = rigidBodyDensity;
                            b.updateInertia();
                        }
                    }
                }
                
                ImGui::End();
            }
            
            ImGui::Render();
            
            drawFrame();
            // We want to animate the particle system using the last frames time to get smooth, frame-rate independent animation
            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            totalTime = glfwGetTime() - start_time;
            lastTime = currentTime;
        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain() {
        vkDestroyImage(device, depthImage, nullptr);
        vkDestroyImageView(device, depthImageView, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, diagnosticPipeline, nullptr);
        vkDestroyPipeline(device, wireframePipeline, nullptr);
        vkDestroyPipeline(device, skyboxPipeline, nullptr);
        vkDestroyPipeline(device, lagrangianPipeline, nullptr);
        vkDestroyPipeline(device, velocityPipeline, nullptr);
        vkDestroyPipeline(device, meshPipeline, nullptr);
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);

        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, initPipeline, nullptr);
        vkDestroyPipeline(device, collideAndStreamPipeline, nullptr);
        vkDestroyPipeline(device, ibmForce1Pipeline, nullptr);
        vkDestroyPipeline(device, ibmForce2Pipeline, nullptr);
        vkDestroyPipeline(device, calcUPipeline, nullptr);
        vkDestroyPipeline(device, updatePositionsPipeline, nullptr);
        vkDestroyPipeline(device, forceReductionPipeline, nullptr);
        vkDestroyPipeline(device, rigidBodySolverPipeline, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyImage(device, skyboxImage, nullptr);
        vkFreeMemory(device, skyboxImageMemory, nullptr);
        vkDestroyImageView(device, skyboxImageView, nullptr);
        vkDestroySampler(device, skyboxSampler, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, renderingUBOBuffers[i], nullptr);
            vkFreeMemory(device, renderingUBOBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, velocityBuffers[i], nullptr);
            vkFreeMemory(device, velocityBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, tempVelBuffers[i], nullptr);
            vkFreeMemory(device, tempVelBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, rhoBuffers[i], nullptr);
            vkFreeMemory(device, rhoBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, flagBuffers[i], nullptr);
            vkFreeMemory(device, flagBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, DDFBuffers[i], nullptr);
            vkFreeMemory(device, DDFBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, borderForceBuffers[i], nullptr);
            vkFreeMemory(device, borderForceBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, lagrangianPointsBuffers[i], nullptr);
            vkFreeMemory(device, lagrangianPointsBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, lagrangianPointsRestBuffers[i], nullptr);
            vkFreeMemory(device, lagrangianPointsRestBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, lagrangianPointsPrevBuffers[i], nullptr);
            vkFreeMemory(device, lagrangianPointsPrevBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, lagrangianDataBuffers[i], nullptr);
            vkFreeMemory(device, lagrangianDataBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, tempForcesBuffers[i], nullptr);
            vkFreeMemory(device, tempForcesBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, skBuffers[i], nullptr);
            vkFreeMemory(device, skBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, totalForceTorqueBuffers[i], nullptr);
            vkFreeMemory(device, totalForceTorqueBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, rigidBodyStateBuffers[i], nullptr);
            vkFreeMemory(device, rigidBodyStateBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, rigidBodyInfoBuffers[i], nullptr);
            vkFreeMemory(device, rigidBodyInfoBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, bodyIndexBuffers[i], nullptr);
            vkFreeMemory(device, bodyIndexBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, wireframeBuffers[i], nullptr);
            vkFreeMemory(device, wireframeBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, wireframeIndexBuffers[i], nullptr);
            vkFreeMemory(device, wireframeIndexBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, skyboxBuffers[i], nullptr);
            vkFreeMemory(device, skyboxBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, skyboxIndexBuffers[i], nullptr);
            vkFreeMemory(device, skyboxIndexBuffersMemory[i], nullptr);
        }
        for (size_t bodyIdx = 0; bodyIdx < rigidBodyCount; ++bodyIdx) {
            for (size_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; ++frameIdx) {
                size_t bufferIdx = bodyIdx * MAX_FRAMES_IN_FLIGHT + frameIdx;
                if (!meshVertexBuffers.empty() && bufferIdx < meshVertexBuffers.size()) {
                    vkDestroyBuffer(device, meshVertexBuffers[bufferIdx], nullptr);
                    vkFreeMemory(device, meshVertexBuffersMemory[bufferIdx], nullptr);
                }
                if (!meshIndexBuffers.empty() && bufferIdx < meshIndexBuffers.size()) {
                    vkDestroyBuffer(device, meshIndexBuffers[bufferIdx], nullptr);
                    vkFreeMemory(device, meshIndexBuffersMemory[bufferIdx], nullptr);
                }
            }
        }

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, graphicsDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, shaderStorageBuffers[i], nullptr);
            vkFreeMemory(device, shaderStorageBuffersMemory[i], nullptr);
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyFence(device, computeInFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures;
        atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomicFloatFeatures.pNext = nullptr;
        atomicFloatFeatures.shaderBufferFloat32Atomics = true; // this allows to perform atomic operations on storage buffers
        atomicFloatFeatures.shaderBufferFloat32AtomicAdd = true; // this allows to perform atomic operations on storage buffers
        atomicFloatFeatures.shaderBufferFloat64Atomics = false;
        atomicFloatFeatures.shaderBufferFloat64AtomicAdd = false;
        atomicFloatFeatures.shaderSharedFloat32Atomics = false;
        atomicFloatFeatures.shaderSharedFloat32AtomicAdd = false;
        atomicFloatFeatures.shaderSharedFloat64Atomics = false;
        atomicFloatFeatures.shaderSharedFloat64AtomicAdd = false;
        atomicFloatFeatures.shaderImageFloat32Atomics = false;
        atomicFloatFeatures.shaderImageFloat32AtomicAdd = false;
        atomicFloatFeatures.sparseImageFloat32Atomics = false;
        atomicFloatFeatures.sparseImageFloat32AtomicAdd = false;
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &atomicFloatFeatures;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsAndComputeFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding lagrangianPointsLayoutBinding{};
        lagrangianPointsLayoutBinding.binding = 2;
        lagrangianPointsLayoutBinding.descriptorCount = 1;
        lagrangianPointsLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lagrangianPointsLayoutBinding.pImmutableSamplers = nullptr;
        lagrangianPointsLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding velocityLayoutBinding{};
        velocityLayoutBinding.binding = 3;
        velocityLayoutBinding.descriptorCount = 1;
        velocityLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        velocityLayoutBinding.pImmutableSamplers = nullptr;
        velocityLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding flagsLayoutBinding{};
        flagsLayoutBinding.binding = 4;
        flagsLayoutBinding.descriptorCount = 1;
        flagsLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        flagsLayoutBinding.pImmutableSamplers = nullptr;
        flagsLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding rigidBodyStateLayoutBinding{};
        rigidBodyStateLayoutBinding.binding = 5;
        rigidBodyStateLayoutBinding.descriptorCount = 1;
        rigidBodyStateLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rigidBodyStateLayoutBinding.pImmutableSamplers = nullptr;
        rigidBodyStateLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        std::array<VkDescriptorSetLayoutBinding, 6> bindings = { uboLayoutBinding, samplerLayoutBinding, lagrangianPointsLayoutBinding, velocityLayoutBinding, flagsLayoutBinding, rigidBodyStateLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &graphicsDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createComputeDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 18> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

        // particles
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // velocity
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // rho
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // flags
        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // DDF
        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[5].pImmutableSamplers = nullptr;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // borderForce
        layoutBindings[6].binding = 6;
        layoutBindings[6].descriptorCount = 1;
        layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[6].pImmutableSamplers = nullptr;
        layoutBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // lagrangianPoints
        layoutBindings[7].binding = 7;
        layoutBindings[7].descriptorCount = 1;
        layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[7].pImmutableSamplers = nullptr;
        layoutBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

        // lagrangianData
        layoutBindings[8].binding = 8;
        layoutBindings[8].descriptorCount = 1;
        layoutBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[8].pImmutableSamplers = nullptr;
        layoutBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // lagrangianPointsRest
        layoutBindings[9].binding = 9;
        layoutBindings[9].descriptorCount = 1;
        layoutBindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[9].pImmutableSamplers = nullptr;
        layoutBindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // lagrangianPointsPrev
        layoutBindings[10].binding = 10;
        layoutBindings[10].descriptorCount = 1;
        layoutBindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[10].pImmutableSamplers = nullptr;
        layoutBindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // tempForces
        layoutBindings[11].binding = 11;
        layoutBindings[11].descriptorCount = 1;
        layoutBindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[11].pImmutableSamplers = nullptr;
        layoutBindings[11].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // totalForceTorque
        layoutBindings[12].binding = 12;
        layoutBindings[12].descriptorCount = 1;
        layoutBindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[12].pImmutableSamplers = nullptr;
        layoutBindings[12].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // rigidBodyState
        layoutBindings[13].binding = 13;
        layoutBindings[13].descriptorCount = 1;
        layoutBindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[13].pImmutableSamplers = nullptr;
        layoutBindings[13].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // s_k buffer
        layoutBindings[14].binding = 14;
        layoutBindings[14].descriptorCount = 1;
        layoutBindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[14].pImmutableSamplers = nullptr;
        layoutBindings[14].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // rigidBodyInfo
        layoutBindings[15].binding = 15;
        layoutBindings[15].descriptorCount = 1;
        layoutBindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[15].pImmutableSamplers = nullptr;
        layoutBindings[15].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // bodyIndex
        layoutBindings[16].binding = 16;
        layoutBindings[16].descriptorCount = 1;
        layoutBindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[16].pImmutableSamplers = nullptr;
        layoutBindings[16].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // tempVel
        layoutBindings[17].binding = 17;
        layoutBindings[17].descriptorCount = 1;
        layoutBindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[17].pImmutableSamplers = nullptr;
        layoutBindings[17].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createGraphicsPipeline() {
        auto particleVertShaderCode = readFile("shaders/particle_vert.spv");
        auto particleFragShaderCode = readFile("shaders/particle_frag.spv");
        VkShaderModule particleVertShaderModule = createShaderModule(particleVertShaderCode);
        VkShaderModule particleFragShaderModule = createShaderModule(particleFragShaderCode);
        VkPipelineShaderStageCreateInfo particleVertShaderStageInfo{};
        particleVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        particleVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        particleVertShaderStageInfo.module = particleVertShaderModule;
        particleVertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo particleFragShaderStageInfo{};
        particleFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        particleFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        particleFragShaderStageInfo.module = particleFragShaderModule;
        particleFragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo particleShaderStages[] = { particleVertShaderStageInfo, particleFragShaderStageInfo };

        auto wireframeVertShaderCode = readFile("shaders/wireframe_vert.spv");
        auto wireframeFragShaderCode = readFile("shaders/wireframe_frag.spv");
        VkShaderModule wireframeVertShaderModule = createShaderModule(wireframeVertShaderCode);
        VkShaderModule wireframeFragShaderModule = createShaderModule(wireframeFragShaderCode);
        VkPipelineShaderStageCreateInfo wireframeVertShaderStageInfo{};
        wireframeVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        wireframeVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        wireframeVertShaderStageInfo.module = wireframeVertShaderModule;
        wireframeVertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo wireframeFragShaderStageInfo{};
        wireframeFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        wireframeFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        wireframeFragShaderStageInfo.module = wireframeFragShaderModule;
        wireframeFragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo wireframeShaderStages[] = { wireframeVertShaderStageInfo, wireframeFragShaderStageInfo };

        auto skyboxVertShaderCode = readFile("shaders/skybox_vert.spv");
        auto skyboxFragShaderCode = readFile("shaders/skybox_frag.spv");
        VkShaderModule skyboxVertShaderModule = createShaderModule(skyboxVertShaderCode);
        VkShaderModule skyboxFragShaderModule = createShaderModule(skyboxFragShaderCode);
        VkPipelineShaderStageCreateInfo skyboxVertShaderStageInfo{};
        skyboxVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        skyboxVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        skyboxVertShaderStageInfo.module = skyboxVertShaderModule;
        skyboxVertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo skyboxFragShaderStageInfo{};
        skyboxFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        skyboxFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        skyboxFragShaderStageInfo.module = skyboxFragShaderModule;
        skyboxFragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo skyboxShaderStages[] = { skyboxVertShaderStageInfo, skyboxFragShaderStageInfo };

        auto meshVertShaderCode = readFile("shaders/mesh_vert.spv");
        auto meshFragShaderCode = readFile("shaders/mesh_frag.spv");
        VkShaderModule meshVertShaderModule = createShaderModule(meshVertShaderCode);
        VkShaderModule meshFragShaderModule = createShaderModule(meshFragShaderCode);
        VkPipelineShaderStageCreateInfo meshVertShaderStageInfo{};
        meshVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        meshVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        meshVertShaderStageInfo.module = meshVertShaderModule;
        meshVertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo meshFragShaderStageInfo{};
        meshFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        meshFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        meshFragShaderStageInfo.module = meshFragShaderModule;
        meshFragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo meshShaderStages[] = { meshVertShaderStageInfo, meshFragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &graphicsDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // particle
        {
            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = particleShaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = graphicsPipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline!");
            }
        }

        // diagnostic particles
        {
            bindingDescription = Particle::getBindingDescription();
            attributeDescriptions = Particle::getAttributeDescriptions();
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkPipelineDepthStencilStateCreateInfo diagnosticDepthStencil{};
            diagnosticDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            diagnosticDepthStencil.depthTestEnable = VK_TRUE;
            diagnosticDepthStencil.depthWriteEnable = VK_TRUE;
            diagnosticDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            diagnosticDepthStencil.depthBoundsTestEnable = VK_FALSE;
            diagnosticDepthStencil.minDepthBounds = 0.0f;
            diagnosticDepthStencil.maxDepthBounds = 1.0f;
            diagnosticDepthStencil.stencilTestEnable = VK_FALSE;
            diagnosticDepthStencil.front = {};
            diagnosticDepthStencil.back = {};

            VkPipelineColorBlendAttachmentState diagnosticColorBlendAttachment{};
            diagnosticColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            diagnosticColorBlendAttachment.blendEnable = VK_FALSE;
            diagnosticColorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            diagnosticColorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            diagnosticColorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            diagnosticColorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            diagnosticColorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            diagnosticColorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

            VkPipelineColorBlendStateCreateInfo diagnosticColorBlending{};
            diagnosticColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            diagnosticColorBlending.logicOpEnable = VK_FALSE;
            diagnosticColorBlending.logicOp = VK_LOGIC_OP_COPY;
            diagnosticColorBlending.attachmentCount = 1;
            diagnosticColorBlending.pAttachments = &diagnosticColorBlendAttachment;
            diagnosticColorBlending.blendConstants[0] = 0.0f;
            diagnosticColorBlending.blendConstants[1] = 0.0f;
            diagnosticColorBlending.blendConstants[2] = 0.0f;
            diagnosticColorBlending.blendConstants[3] = 0.0f;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = particleShaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &diagnosticDepthStencil;
            pipelineInfo.pColorBlendState = &diagnosticColorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = graphicsPipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = graphicsPipeline;
            pipelineInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &diagnosticPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create diagnostic pipeline!");
            }
        }

        // wireframe
        {
            bindingDescription = Vertex::getBindingDescription();
            attributeDescriptions = Vertex::getAttributeDescriptions();
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

            VkPipelineDepthStencilStateCreateInfo wireframeDepthStencil{};
            wireframeDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            wireframeDepthStencil.depthTestEnable = VK_TRUE;
            wireframeDepthStencil.depthWriteEnable = VK_TRUE;
            wireframeDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            wireframeDepthStencil.depthBoundsTestEnable = VK_FALSE;
            wireframeDepthStencil.minDepthBounds = 0.0f;
            wireframeDepthStencil.maxDepthBounds = 1.0f;
            wireframeDepthStencil.stencilTestEnable = VK_FALSE;
            wireframeDepthStencil.front = {};
            wireframeDepthStencil.back = {};

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = wireframeShaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &wireframeDepthStencil;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = graphicsPipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = graphicsPipeline;
            pipelineInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &wireframePipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline!");
            }
        }

        // mesh
        {
            auto meshBindingDescription = MeshVertex::getBindingDescription();
            auto meshAttributeDescriptions = MeshVertex::getAttributeDescriptions();
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(meshAttributeDescriptions.size());
            vertexInputInfo.pVertexBindingDescriptions = &meshBindingDescription;
            vertexInputInfo.pVertexAttributeDescriptions = meshAttributeDescriptions.data();
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo meshRasterizer{};
            meshRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            meshRasterizer.depthClampEnable = VK_FALSE;
            meshRasterizer.rasterizerDiscardEnable = VK_FALSE;
            meshRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            meshRasterizer.lineWidth = 1.0f;
            meshRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            meshRasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
            meshRasterizer.depthBiasEnable = VK_FALSE;

            VkPipelineDepthStencilStateCreateInfo meshDepthStencil{};
            meshDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            meshDepthStencil.depthTestEnable = VK_TRUE;
            meshDepthStencil.depthWriteEnable = VK_TRUE;
            meshDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            meshDepthStencil.depthBoundsTestEnable = VK_FALSE;
            meshDepthStencil.minDepthBounds = 0.0f;
            meshDepthStencil.maxDepthBounds = 1.0f;
            meshDepthStencil.stencilTestEnable = VK_FALSE;
            meshDepthStencil.front = {};
            meshDepthStencil.back = {};

            VkPipelineColorBlendAttachmentState meshColorBlendAttachment{};
            meshColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            meshColorBlendAttachment.blendEnable = VK_FALSE;
            meshColorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            meshColorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            meshColorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            meshColorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            meshColorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            meshColorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

            VkPipelineColorBlendStateCreateInfo meshColorBlending{};
            meshColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            meshColorBlending.logicOpEnable = VK_FALSE;
            meshColorBlending.logicOp = VK_LOGIC_OP_COPY;
            meshColorBlending.attachmentCount = 1;
            meshColorBlending.pAttachments = &meshColorBlendAttachment;
            meshColorBlending.blendConstants[0] = 0.0f;
            meshColorBlending.blendConstants[1] = 0.0f;
            meshColorBlending.blendConstants[2] = 0.0f;
            meshColorBlending.blendConstants[3] = 0.0f;

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = meshShaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &meshRasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &meshDepthStencil;
            pipelineInfo.pColorBlendState = &meshColorBlending;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = graphicsPipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = graphicsPipeline;
            pipelineInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create mesh pipeline!");
            }
        }

        //// skybox
        //{
        //    VkVertexInputBindingDescription bindingDescription{};
        //    bindingDescription.binding = 0;
        //    bindingDescription.stride = sizeof(glm::vec3);
        //    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        //
        //    std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
        //    attributeDescriptions[0].binding = 0;
        //    attributeDescriptions[0].location = 0;
        //    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        //    attributeDescriptions[0].offset = 0;
        //
        //    vertexInputInfo.vertexBindingDescriptionCount = 1;
        //    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        //    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        //    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        //    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        //
        //    depthStencil.depthTestEnable = VK_FALSE;
        //    depthStencil.depthWriteEnable = VK_FALSE;
        //    depthStencil.depthBoundsTestEnable = VK_FALSE;
        //    depthStencil.stencilTestEnable = VK_FALSE;
        //
        //    VkGraphicsPipelineCreateInfo pipelineInfo{};
        //    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        //    pipelineInfo.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT;
        //    pipelineInfo.stageCount = 2;
        //    pipelineInfo.pStages = skyboxShaderStages;
        //    pipelineInfo.pVertexInputState = &vertexInputInfo;
        //    pipelineInfo.pInputAssemblyState = &inputAssembly;
        //    pipelineInfo.pViewportState = &viewportState;
        //    pipelineInfo.pRasterizationState = &rasterizer;
        //    pipelineInfo.pMultisampleState = &multisampling;
        //    pipelineInfo.pDepthStencilState = &depthStencil;
        //    pipelineInfo.pColorBlendState = &colorBlending;
        //    pipelineInfo.pDynamicState = &dynamicState;
        //    pipelineInfo.layout = graphicsPipelineLayout;
        //    pipelineInfo.renderPass = renderPass;
        //    pipelineInfo.subpass = 0;
        //    pipelineInfo.basePipelineHandle = graphicsPipeline;
        //    pipelineInfo.basePipelineIndex = -1;
        //
        //    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyboxPipeline) != VK_SUCCESS) {
        //        throw std::runtime_error("failed to create graphics pipeline!");
        //    }
        //}

        vkDestroyShaderModule(device, particleFragShaderModule, nullptr);
        vkDestroyShaderModule(device, particleVertShaderModule, nullptr);
        vkDestroyShaderModule(device, wireframeFragShaderModule, nullptr);
        vkDestroyShaderModule(device, wireframeVertShaderModule, nullptr);
        vkDestroyShaderModule(device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(device, meshFragShaderModule, nullptr);
        vkDestroyShaderModule(device, meshVertShaderModule, nullptr);
    }

    void createSkyBoxPipeline() {
        auto vertShaderCode = readFile("shaders/skybox_vert.spv");
        auto fragShaderCode = readFile("shaders/skybox_frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        //auto bindingDescription = Vertex::getBindingDescription();
        //auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(glm::vec3);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = 0;

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyboxPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createLagrangianPipeline() {
        auto vertShaderCode = readFile("shaders/lagrangian_vert.spv");
        auto fragShaderCode = readFile("shaders/lagrangian_frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &lagrangianPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create lagrangian pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createVelocityPipeline() {
        auto vertShaderCode = readFile("shaders/velocity_vert.spv");
        auto fragShaderCode = readFile("shaders/velocity_frag.spv");
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &velocityPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create velocity pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createComputePipeline() {
        {
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;

            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline layout!");
            }
        }

        {
            auto computeShaderCode = readFile("shaders/init_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &initPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/collide_and_stream_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &collideAndStreamPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/ibm_force1_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ibmForce1Pipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/ibm_force2_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ibmForce2Pipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/calc_u_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &calcUPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/update_positions_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &updatePositionsPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/force_reduction_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &forceReductionPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/rigid_body_solver_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rigidBodySolverPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }

        {
            auto computeShaderCode = readFile("shaders/calc_comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
            computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStageInfo.module = computeShaderModule;
            computeShaderStageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline!");
            }

            vkDestroyShaderModule(device, computeShaderModule, nullptr);
        }
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        //VkImageSubresourceRange subresourceRange{};
        //subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        //subresourceRange.baseMipLevel = 0;
        //subresourceRange.levelCount = 1;
        //subresourceRange.baseArrayLayer = 0;
        //transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, subresourceRange);
    }

    void createSkybox() {
        ktxResult result;
        ktxTexture* ktxTexture;

        result = ktxTexture_CreateFromNamedFile("textures/cubemap_space.ktx", KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
        assert(result == KTX_SUCCESS, "cannot load skybox texture");

        const uint32_t width = ktxTexture->baseWidth;
        const uint32_t height = ktxTexture->baseHeight;
        const uint32_t mipLevels = ktxTexture->numLevels;

        ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
        ktx_size_t ktxTextureSize = ktxTexture_GetDataSize(ktxTexture);

        VkMemoryAllocateInfo memAllocInfo{};
        memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        VkMemoryRequirements memReqs;

        // Create a host-visible staging buffer that contains the raw image data
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = ktxTextureSize;
        // This buffer is used as a transfer source for the buffer copy
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create staging buffer!");
        }

        // Get memory requirements for the staging buffer (alignment, memory type bits)
        vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);
        memAllocInfo.allocationSize = memReqs.size;
        // Get memory type index for a host visible buffer
        memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate staging memory!");
        }
        if (vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("failed to bind staging buffer memory!");
        }

        // Copy texture data into staging buffer
        uint8_t* data;
        vkMapMemory(device, stagingMemory, 0, memReqs.size, 0, (void**)&data);
        memcpy(data, ktxTextureData, ktxTextureSize);
        vkUnmapMemory(device, stagingMemory);

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        // Create optimal tiled target image
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = mipLevels;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.extent = { width, height, 1 };
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        // Cube faces count as array layers in Vulkan
        imageCreateInfo.arrayLayers = 6;
        // This flag is required for cube map images
        imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        if (vkCreateImage(device, &imageCreateInfo, nullptr, &skyboxImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox image!");
        }

        vkGetImageMemoryRequirements(device, skyboxImage, &memReqs);
        memAllocInfo.allocationSize = memReqs.size;
        memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &memAllocInfo, nullptr, &skyboxImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }
        if (vkBindImageMemory(device, skyboxImage, skyboxImageMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("failed to bind image memory!");
        }

        VkCommandBuffer copyCmd = beginSingleTimeCommands();// Setup buffer copy regions for each face including all of its miplevels
        std::vector<VkBufferImageCopy> bufferCopyRegions;
        uint32_t offset = 0;

        for (uint32_t face = 0; face < 6; face++)
        {
            for (uint32_t level = 0; level < mipLevels; level++)
            {
                // Calculate offset into staging buffer for the current mip level and face
                ktx_size_t offset;
                KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
                assert(ret == KTX_SUCCESS);
                VkBufferImageCopy bufferCopyRegion = {};
                bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                bufferCopyRegion.imageSubresource.mipLevel = level;
                bufferCopyRegion.imageSubresource.baseArrayLayer = face;
                bufferCopyRegion.imageSubresource.layerCount = 1;
                bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
                bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
                bufferCopyRegion.imageExtent.depth = 1;
                bufferCopyRegion.bufferOffset = offset;
                bufferCopyRegions.push_back(bufferCopyRegion);
            }
        }

        // Image barrier for optimal image (target)
        // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = mipLevels;
        subresourceRange.layerCount = 6;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = skyboxImage;
        barrier.subresourceRange = subresourceRange;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(
            copyCmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        // Copy the cube map faces from the staging buffer to the optimal tiled image
        vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            skyboxImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(bufferCopyRegions.size()),
            bufferCopyRegions.data()
        );

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            copyCmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(copyCmd);

        // Create sampler
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.compareOp = VK_COMPARE_OP_NEVER;
        sampler.minLod = 0.0f;
        sampler.maxLod = static_cast<float>(mipLevels);
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler.maxAnisotropy = 1.0f;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        sampler.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        sampler.anisotropyEnable = VK_FALSE;
        if (vkCreateSampler(device, &sampler, nullptr, &skyboxSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox sampler!");
        }

        // Create image view
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        // Cube map view type
        view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view.format = format;
        view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        // 6 array layers (faces)
        view.subresourceRange.layerCount = 6;
        // Set number of mip levels
        view.subresourceRange.levelCount = mipLevels;
        view.image = skyboxImage;
        if (vkCreateImageView(device, &view, nullptr, &skyboxImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox image view!");
        }

        // Clean up staging resources
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        ktxTexture_Destroy(ktxTexture);
    }

    void createVertexBuffers() {
        // wireframe
        {
            VkDeviceSize bufferSize = sizeof(wireframeVertices[0]) * wireframeVertices.size();
            wireframeBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            wireframeBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, wireframeVertices.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wireframeBuffers[i], wireframeBuffersMemory[i]);
                copyBuffer(stagingBuffer, wireframeBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // skybox
        {
            VkDeviceSize bufferSize = sizeof(skyboxVertices[0]) * skyboxVertices.size();
            skyboxBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            skyboxBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, skyboxVertices.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxBuffers[i], skyboxBuffersMemory[i]);
                copyBuffer(stagingBuffer, skyboxBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // Lagrangian Points - generate based on shape
        {
            if (rigidBodies.empty()) {
                RigidBody body1;
                body1.shape = RigidBodyShape::MESH;
                body1.meshFilePath = "models/fan_pointcloud.glb";
                body1.meshScale = 20.0f;
                body1.meshCoordSystem = 0;
                body1.rho = 1.0f;
                body1.position = glm::vec3(Nx / 2.0f - 30.0f, Ny / 2.0f, Nz / 2.0f);
                body1.orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                body1.linear_velocity = glm::vec3(0.0f);
                body1.angular_velocity = glm::vec3(0.0f, 0.02f, 0.0f);
                body1.isManualControl = false;
                body1.manualLinVel[0] = 0.0f;
                body1.manualLinVel[1] = 0.0f;
                body1.manualLinVel[2] = 0.0f;
                body1.manualAngVel[0] = 0.0f;
                body1.manualAngVel[1] = 0.02f;
                body1.manualAngVel[2] = 0.0f;
                // rigidBodies.push_back(body1);

                RigidBody body2;
                body2.shape = RigidBodyShape::MESH;
                body2.meshFilePath = "models/sphere_pointcloud.glb";
                body2.meshScale = 10.0f;
                body2.meshCoordSystem = 0;
                body2.rho = 1.0f;
                body2.position = glm::vec3(Nx / 2.0f + 30.0f, Ny / 2.0f, Nz / 2.0f);
                body2.orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                body2.linear_velocity = glm::vec3(0.0f);
                body2.angular_velocity = glm::vec3(0.0f, 0.0f, -0.02f);
                body2.isManualControl = false;
                body2.manualLinVel[0] = 0.0f;
                body2.manualLinVel[1] = 0.0f;
                body2.manualLinVel[2] = 0.0f;
                body2.manualAngVel[0] = 0.0f;
                body2.manualAngVel[1] = 0.0f;
                body2.manualAngVel[2] = -0.02f;
                body2.updateInertia();
                rigidBodies.push_back(body2);
            }
            rigidBodyCount = static_cast<uint32_t>(rigidBodies.size());
            
            std::vector<glm::vec3> positions;
            std::vector<float> skValues;
            std::vector<glm::vec3> normals;
            std::vector<uint32_t> bodyIndices;
            
            float target_spacing = 0.5f;
            
            for (uint32_t bodyIdx = 0; bodyIdx < rigidBodyCount; bodyIdx++) {
                const RigidBody& body = rigidBodies[bodyIdx];
                size_t startIdx = positions.size();
                
                switch (body.shape) {
                    case RigidBodyShape::SPHERE:
                        generateSphere(body.radius, target_spacing, positions, skValues);
                        break;
                    case RigidBodyShape::BOX:
                        generateBox(body.boxSize, target_spacing, positions, skValues);
                        break;
                    case RigidBodyShape::CYLINDER:
                        generateCylinder(body.cylinderRadius, body.cylinderHeight, target_spacing, positions, skValues);
                        break;
                    case RigidBodyShape::MESH:
                        {
                            std::string meshPath = body.meshFilePath;
                            
                            bool loaded = false;
                            std::vector<glm::vec3> meshPositions;
                            std::vector<float> meshSkValues;
                            std::vector<glm::vec3> meshNormals;
                            
                            if (meshPath.find(".glb") != std::string::npos || meshPath.find(".GLB") != std::string::npos) {
                                loaded = loadMeshFromGLB(meshPath, meshPositions, meshSkValues, meshNormals);
                            } else if (meshPath.find(".bin") != std::string::npos || meshPath.find(".BIN") != std::string::npos) {
                                loaded = loadMeshFromBIN(meshPath, meshPositions, meshSkValues, meshNormals);
                            } else {
                                std::string csvPath = meshPath;
                                size_t dotPos = meshPath.find_last_of('.');
                                if (dotPos != std::string::npos) {
                                    csvPath = meshPath.substr(0, dotPos) + ".csv";
                                }
                                loaded = loadMeshFromCSV(csvPath, meshPositions, meshSkValues, meshNormals);
                            }
                            
                            if (!loaded || meshPositions.empty()) {
                                std::cerr << "Failed to load mesh data for body " << bodyIdx << ", falling back to BOX shape" << std::endl;
                                generateBox(body.boxSize, target_spacing, meshPositions, meshSkValues);
                            } else {
                                if (body.meshCoordSystem == 1) {
                                    std::cout << "Applying coordinate transform: Y-up (OpenGL) -> Z-up (Vulkan)" << std::endl;
                                    for (auto& pos : meshPositions) {
                                        float tmp = pos.y;
                                        pos.y = pos.z;
                                        pos.z = tmp;
                                    }
                                    for (auto& norm : meshNormals) {
                                        float tmp = norm.y;
                                        norm.y = norm.z;
                                        norm.z = tmp;
                                    }
                                } else if (body.meshCoordSystem == 2) {
                                    std::cout << "Applying coordinate transform: Z-up -> Y-up" << std::endl;
                                    for (auto& pos : meshPositions) {
                                        float tmp = pos.z;
                                        pos.z = pos.y;
                                        pos.y = tmp;
                                    }
                                    for (auto& norm : meshNormals) {
                                        float tmp = norm.z;
                                        norm.z = norm.y;
                                        norm.y = tmp;
                                    }
                                }
                                
                                float scale = body.meshScale;
                                for (auto& pos : meshPositions) {
                                    pos *= scale;
                                }
                                for (auto& sk : meshSkValues) {
                                    sk *= scale * scale;
                                }
                                
                                rigidBodies[bodyIdx].meshBoundingBoxMin = meshPositions[0];
                                rigidBodies[bodyIdx].meshBoundingBoxMax = meshPositions[0];
                                for (const auto& pos : meshPositions) {
                                    rigidBodies[bodyIdx].meshBoundingBoxMin = glm::min(rigidBodies[bodyIdx].meshBoundingBoxMin, pos);
                                    rigidBodies[bodyIdx].meshBoundingBoxMax = glm::max(rigidBodies[bodyIdx].meshBoundingBoxMax, pos);
                                }
                                std::cout << "Body " << bodyIdx << " mesh bounding box (scaled): ("
                                          << rigidBodies[bodyIdx].meshBoundingBoxMin.x << ", " 
                                          << rigidBodies[bodyIdx].meshBoundingBoxMin.y << ", " 
                                          << rigidBodies[bodyIdx].meshBoundingBoxMin.z << ") to ("
                                          << rigidBodies[bodyIdx].meshBoundingBoxMax.x << ", " 
                                          << rigidBodies[bodyIdx].meshBoundingBoxMax.y << ", " 
                                          << rigidBodies[bodyIdx].meshBoundingBoxMax.z << ")" << std::endl;
                                float totalArea = 0.0f;
                                for (float sk : meshSkValues) totalArea += sk;
                                std::cout << "Total surface area from sk values (scaled): " << totalArea << std::endl;
                                rigidBodies[bodyIdx].updateInertia();
                            }
                            
                            positions.insert(positions.end(), meshPositions.begin(), meshPositions.end());
                            skValues.insert(skValues.end(), meshSkValues.begin(), meshSkValues.end());
                        }
                        break;
                }
                
                size_t endIdx = positions.size();
                for (size_t i = startIdx; i < endIdx; i++) {
                    bodyIndices.push_back(bodyIdx);
                }
            }
            
            for (uint32_t bodyIdx = 0; bodyIdx < rigidBodyCount; bodyIdx++) {
                if (rigidBodies[bodyIdx].shape == RigidBodyShape::MESH) {
                    size_t startIdx = 0;
                    for (uint32_t j = 0; j < bodyIdx; j++) {
                        RigidBody& prevBody = rigidBodies[j];
                        switch (prevBody.shape) {
                            case RigidBodyShape::SPHERE:
                                { std::vector<glm::vec3> tmpPos; std::vector<float> tmpSk; generateSphere(prevBody.radius, target_spacing, tmpPos, tmpSk); startIdx += tmpPos.size(); }
                                break;
                            case RigidBodyShape::BOX:
                                { std::vector<glm::vec3> tmpPos; std::vector<float> tmpSk; generateBox(prevBody.boxSize, target_spacing, tmpPos, tmpSk); startIdx += tmpPos.size(); }
                                break;
                            case RigidBodyShape::CYLINDER:
                                { std::vector<glm::vec3> tmpPos; std::vector<float> tmpSk; generateCylinder(prevBody.cylinderRadius, prevBody.cylinderHeight, target_spacing, tmpPos, tmpSk); startIdx += tmpPos.size(); }
                                break;
                            case RigidBodyShape::MESH:
                                break;
                        }
                    }
                    bool hasPoints = false;
                    for (size_t i = startIdx; i < bodyIndices.size() && bodyIndices[i] == bodyIdx; i++) {
                        hasPoints = true;
                        break;
                    }
                    if (!hasPoints) {
                        rigidBodies[bodyIdx].shape = RigidBodyShape::BOX;
                        rigidBodies[bodyIdx].updateInertia();
                    }
                }
            }
            
            lagrangianPointCount = static_cast<uint32_t>(positions.size());
            
            std::vector<LagrangianPoint> lagrangianPointsRest(lagrangianPointCount);
            std::vector<LagrangianData> lagrangianData(lagrangianPointCount);

            for (uint32_t i = 0; i < lagrangianPointCount; i++) {
                lagrangianPointsRest[i].position = glm::vec4(positions[i], 1.0f);
                lagrangianData[i].velocity = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                lagrangianData[i].force = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            VkDeviceSize pointsBufferSize = sizeof(LagrangianPoint) * lagrangianPointCount;
            VkDeviceSize dataBufferSize = sizeof(LagrangianData) * lagrangianPointCount;

            lagrangianPointsBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianPointsBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianPointsRestBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianPointsRestBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianPointsPrevBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianPointsPrevBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianDataBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            lagrangianDataBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            void* data;

            createBuffer(pointsBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, pointsBufferSize, 0, &data);
            memcpy(data, lagrangianPointsRest.data(), (size_t)pointsBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(pointsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lagrangianPointsBuffers[i], lagrangianPointsBuffersMemory[i]);
                copyBuffer(stagingBuffer, lagrangianPointsBuffers[i], pointsBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            createBuffer(pointsBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, pointsBufferSize, 0, &data);
            memcpy(data, lagrangianPointsRest.data(), (size_t)pointsBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(pointsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lagrangianPointsRestBuffers[i], lagrangianPointsRestBuffersMemory[i]);
                copyBuffer(stagingBuffer, lagrangianPointsRestBuffers[i], pointsBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            createBuffer(pointsBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, pointsBufferSize, 0, &data);
            memcpy(data, lagrangianPointsRest.data(), (size_t)pointsBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(pointsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lagrangianPointsPrevBuffers[i], lagrangianPointsPrevBuffersMemory[i]);
                copyBuffer(stagingBuffer, lagrangianPointsPrevBuffers[i], pointsBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            createBuffer(dataBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, dataBufferSize, 0, &data);
            memcpy(data, lagrangianData.data(), (size_t)dataBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(dataBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lagrangianDataBuffers[i], lagrangianDataBuffersMemory[i]);
                copyBuffer(stagingBuffer, lagrangianDataBuffers[i], dataBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            VkDeviceSize tempForceBufferSize = sizeof(glm::vec4) * lagrangianPointCount;
            tempForcesBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            tempForcesBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(tempForceBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tempForcesBuffers[i], tempForcesBuffersMemory[i]);
            }

            VkDeviceSize skBufferSize = sizeof(float) * lagrangianPointCount;

            skBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            skBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            createBuffer(skBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, skBufferSize, 0, &data);
            memcpy(data, skValues.data(), (size_t)skBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(skBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skBuffers[i], skBuffersMemory[i]);
                copyBuffer(stagingBuffer, skBuffers[i], skBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            VkDeviceSize totalForceTorqueBufferSize = sizeof(glm::vec4) * 2 * rigidBodyCount;
            totalForceTorqueBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            totalForceTorqueBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
            totalForceTorqueBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<glm::vec4> initialForceTorque(rigidBodyCount * 2, glm::vec4(0.0f));

            createBuffer(totalForceTorqueBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, totalForceTorqueBufferSize, 0, &data);
            memcpy(data, initialForceTorque.data(), (size_t)totalForceTorqueBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(totalForceTorqueBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, totalForceTorqueBuffers[i], totalForceTorqueBuffersMemory[i]);
                copyBuffer(stagingBuffer, totalForceTorqueBuffers[i], totalForceTorqueBufferSize);
                vkMapMemory(device, totalForceTorqueBuffersMemory[i], 0, totalForceTorqueBufferSize, 0, &totalForceTorqueBuffersMapped[i]);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            VkDeviceSize rigidBodyStateBufferSize = sizeof(RigidBodyState) * rigidBodyCount;
            rigidBodyStateBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            rigidBodyStateBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<RigidBodyState> initialRigidBodyStates(rigidBodyCount);
            for (uint32_t i = 0; i < rigidBodyCount; i++) {
                initialRigidBodyStates[i] = rigidBodies[i].getRigidBodyState();
            }

            createBuffer(rigidBodyStateBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, rigidBodyStateBufferSize, 0, &data);
            memcpy(data, initialRigidBodyStates.data(), (size_t)rigidBodyStateBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(rigidBodyStateBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rigidBodyStateBuffers[i], rigidBodyStateBuffersMemory[i]);
                copyBuffer(stagingBuffer, rigidBodyStateBuffers[i], rigidBodyStateBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            VkDeviceSize rigidBodyInfoBufferSize = sizeof(RigidBodyInfo) * rigidBodyCount;
            rigidBodyInfoBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            rigidBodyInfoBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<RigidBodyInfo> rigidBodyInfos(rigidBodyCount);
            for (uint32_t i = 0; i < rigidBodyCount; i++) {
                rigidBodyInfos[i] = rigidBodies[i].getRigidBodyInfo();
            }

            createBuffer(rigidBodyInfoBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, rigidBodyInfoBufferSize, 0, &data);
            memcpy(data, rigidBodyInfos.data(), (size_t)rigidBodyInfoBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(rigidBodyInfoBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rigidBodyInfoBuffers[i], rigidBodyInfoBuffersMemory[i]);
                copyBuffer(stagingBuffer, rigidBodyInfoBuffers[i], rigidBodyInfoBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

            VkDeviceSize bodyIndexBufferSize = sizeof(uint32_t) * lagrangianPointCount;
            bodyIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            bodyIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            createBuffer(bodyIndexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            vkMapMemory(device, stagingBufferMemory, 0, bodyIndexBufferSize, 0, &data);
            memcpy(data, bodyIndices.data(), (size_t)bodyIndexBufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bodyIndexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bodyIndexBuffers[i], bodyIndexBuffersMemory[i]);
                copyBuffer(stagingBuffer, bodyIndexBuffers[i], bodyIndexBufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // mesh - load mesh for each rigid body
        {
            meshVerticesPerBody.resize(rigidBodyCount);
            meshIndicesPerBody.resize(rigidBodyCount);
            meshVertexBuffers.resize(rigidBodyCount * MAX_FRAMES_IN_FLIGHT);
            meshVertexBuffersMemory.resize(rigidBodyCount * MAX_FRAMES_IN_FLIGHT);
            meshIndexBuffers.resize(rigidBodyCount * MAX_FRAMES_IN_FLIGHT);
            meshIndexBuffersMemory.resize(rigidBodyCount * MAX_FRAMES_IN_FLIGHT);
            
            for (size_t bodyIdx = 0; bodyIdx < rigidBodyCount; ++bodyIdx) {
                std::string meshPath = rigidBodies[bodyIdx].meshFilePath;
                size_t pos = meshPath.find("_pointcloud.glb");
                if (pos != std::string::npos) {
                    meshPath = meshPath.substr(0, pos) + ".glb";
                }
                
                std::cout << "Loading mesh for body " << bodyIdx << ": " << meshPath << std::endl;
                
                if (loadMeshTrianglesFromGLB(meshPath, meshVerticesPerBody[bodyIdx], meshIndicesPerBody[bodyIdx])) {
                    float scale = rigidBodies[bodyIdx].meshScale;
                    int coordSystem = rigidBodies[bodyIdx].meshCoordSystem;
                    
                    for (auto& vertex : meshVerticesPerBody[bodyIdx]) {
                        vertex.pos *= scale;
                        
                        if (coordSystem == 0) {
                            float tmp = vertex.pos.y;
                            vertex.pos.y = vertex.pos.z;
                            vertex.pos.z = tmp;
                            
                            tmp = vertex.normal.y;
                            vertex.normal.y = vertex.normal.z;
                            vertex.normal.z = tmp;
                        }
                    }
                    
                    std::cout << "  Loaded " << meshVerticesPerBody[bodyIdx].size() << " vertices, " 
                              << meshIndicesPerBody[bodyIdx].size() << " indices for body " << bodyIdx << std::endl;
                } else {
                    std::cerr << "  Failed to load mesh for body " << bodyIdx << std::endl;
                }
            }
            
            // Create vertex buffers for each rigid body
            for (size_t bodyIdx = 0; bodyIdx < rigidBodyCount; ++bodyIdx) {
                if (!meshVerticesPerBody[bodyIdx].empty()) {
                    VkDeviceSize bufferSize = sizeof(MeshVertex) * meshVerticesPerBody[bodyIdx].size();
                    
                    VkBuffer stagingBuffer;
                    VkDeviceMemory stagingBufferMemory;
                    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

                    void* data;
                    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
                    memcpy(data, meshVerticesPerBody[bodyIdx].data(), (size_t)bufferSize);
                    vkUnmapMemory(device, stagingBufferMemory);

                    for (size_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; frameIdx++) {
                        size_t bufferIdx = bodyIdx * MAX_FRAMES_IN_FLIGHT + frameIdx;
                        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, meshVertexBuffers[bufferIdx], meshVertexBuffersMemory[bufferIdx]);
                        copyBuffer(stagingBuffer, meshVertexBuffers[bufferIdx], bufferSize);
                    }

                    vkDestroyBuffer(device, stagingBuffer, nullptr);
                    vkFreeMemory(device, stagingBufferMemory, nullptr);
                }
            }
        }
    }

    void createIndexBuffers() {
        // wireframe
        {
            VkDeviceSize bufferSize = sizeof(wireframeIndices[0]) * wireframeIndices.size();
            wireframeIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            wireframeIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, wireframeIndices.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wireframeIndexBuffers[i], wireframeIndexBuffersMemory[i]);
                copyBuffer(stagingBuffer, wireframeIndexBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }


        // skybox
        {
            VkDeviceSize bufferSize = sizeof(skyboxIndices[0]) * skyboxIndices.size();
            skyboxIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            skyboxIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, skyboxIndices.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxIndexBuffers[i], skyboxIndexBuffersMemory[i]);
                copyBuffer(stagingBuffer, skyboxIndexBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // mesh - create index buffers for each rigid body
        {
            for (size_t bodyIdx = 0; bodyIdx < rigidBodyCount; ++bodyIdx) {
                if (!meshIndicesPerBody[bodyIdx].empty()) {
                    VkDeviceSize bufferSize = sizeof(uint32_t) * meshIndicesPerBody[bodyIdx].size();
                    
                    VkBuffer stagingBuffer;
                    VkDeviceMemory stagingBufferMemory;
                    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

                    void* data;
                    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
                    memcpy(data, meshIndicesPerBody[bodyIdx].data(), (size_t)bufferSize);
                    vkUnmapMemory(device, stagingBufferMemory);

                    for (size_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; frameIdx++) {
                        size_t bufferIdx = bodyIdx * MAX_FRAMES_IN_FLIGHT + frameIdx;
                        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, meshIndexBuffers[bufferIdx], meshIndexBuffersMemory[bufferIdx]);
                        copyBuffer(stagingBuffer, meshIndexBuffers[bufferIdx], bufferSize);
                    }

                    vkDestroyBuffer(device, stagingBuffer, nullptr);
                    vkFreeMemory(device, stagingBufferMemory, nullptr);
                }
            }
        }
    }

    void createShaderStorageBuffers() {
        // particles
        {
            // Initialize particles
            std::default_random_engine rndEngine((unsigned)time(nullptr));
            std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
            std::uniform_real_distribution<float> rndLife(8.0f, 20.0f);

            particle_count = 20000000;
            std::vector<Particle> particles(particle_count);

            for (uint32_t idx = 0; idx < particle_count; ++idx) {
                Particle& particle = particles[idx];
                particle.position = { 
                    2.0f + rndDist(rndEngine) * (Nx - 4.0f), 
                    2.0f + rndDist(rndEngine) * (Ny - 4.0f), 
                    2.0f + rndDist(rndEngine) * (Nz / 2 - 2.0f), 
                    0.0f
                };
                particle.color = { 0.0f, 1.0f, 0.0f, (useEmitter == 1) ? 0.0f : rndLife(rndEngine) };
            }

            VkDeviceSize bufferSize = sizeof(Particle) * particle_count;

            // Create a staging buffer used to upload data to the gpu
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, particles.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            // Copy initial particle data to all storage buffers
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBuffers[i], shaderStorageBuffersMemory[i]);
                copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // velocity
        {
            // create velocity buffer, size = width * height * (vec2)
            VkDeviceSize bufferSize = Nxyz * sizeof(float) * 3;
            velocityBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            velocityBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<float> vels(Nxyz * 3, 0.0f);
            parallel_for(Nxyz, [&](uint32_t index) { uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;
                if (z == 0) {
                    float cx = Nx / 2.0f, cy = Ny / 2.0f;
                    float dx = x - cx, dy = y - cy;
                    float dist = sqrt(dx * dx + dy * dy);
                    float maxDist = sqrt(cx * cx + cy * cy);
                    float t = std::min(dist / maxDist, 1.0f);
                    // vels[index + 2 * Nxyz] = 0.05f * (1.0f - t) + 0.05f * t;
                }
                else if (z == Nz - 1) {
                    // vels[index + 2 * Nxyz] = 0.0f;
                }
            });

            // create staging buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, vels.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // create velocity buffer
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityBuffers[i], velocityBuffersMemory[i]);
                copyBuffer(stagingBuffer, velocityBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // tempVel
        {
            VkDeviceSize bufferSize = Nxyz * sizeof(float) * 3;
            tempVelBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            tempVelBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tempVelBuffers[i], tempVelBuffersMemory[i]);
            }
        }

        // rho
        {
            // create rho buffer, size = width * height * (float)
            VkDeviceSize bufferSize = Nxyz * sizeof(float);
            rhoBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            rhoBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<float> rho(Nxyz, 1.0f);

            // create staging buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, rho.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // create rho buffer
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rhoBuffers[i], rhoBuffersMemory[i]);
                copyBuffer(stagingBuffer, rhoBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // flags
        {
            auto cube = [](const uint x, const uint y, const uint z, const glm::vec3& p, const float l) {
                const glm::vec3 t = glm::vec3(x, y, z) - p;
                return t.x >= -0.5f * l && t.x <= 0.5f * l && t.y >= -0.5f * l && t.y <= 0.5f * l && t.z >= -0.5f * l && t.z <= 0.5f * l;
            };

            // create flags buffer, size = width * height * (float)
            VkDeviceSize bufferSize = Nxyz * sizeof(uint32_t);
            flagBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            flagBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            std::vector<uint> flags(Nxyz, 0);
            parallel_for(Nxyz, [&](uint32_t index) { uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;
                if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) {
                    flags[index] = TYPE_X;
                } else if (z == 0) {
                    flags[index] = TYPE_X;
                } else if (z == Nz - 1) {
                    flags[index] = TYPE_X;
                }
            });

            // create staging buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, flags.data(), (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // create flags buffer
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, flagBuffers[i], flagBuffersMemory[i]);
                copyBuffer(stagingBuffer, flagBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // DDF
        {
            // create FOld buffer, size = width * height * (vecQ)
            VkDeviceSize bufferSize = Nxyz * sizeof(float) * Q;

            DDFBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            DDFBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            //create staging buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memset(data, 0, (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // create velocity buffer
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DDFBuffers[i], DDFBuffersMemory[i]);
                copyBuffer(stagingBuffer, DDFBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        // borderForce
        {
            VkDeviceSize bufferSize = Nxyz * sizeof(float) * 3;

            borderForceBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            borderForceBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

            //create staging buffer
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memset(data, 0, (size_t)bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // create velocity buffer
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, borderForceBuffers[i], borderForceBuffersMemory[i]);
                copyBuffer(stagingBuffer, borderForceBuffers[i], bufferSize);
            }

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize1 = sizeof(SimulateUBO);
        VkDeviceSize bufferSize2 = sizeof(RenderingUBO);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        renderingUBOBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        renderingUBOBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        renderingUBOBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize1, 0, &uniformBuffersMapped[i]);
            createBuffer(bufferSize2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, renderingUBOBuffers[i], renderingUBOBuffersMemory[i]);
            vkMapMemory(device, renderingUBOBuffersMemory[i], 0, bufferSize2, 0, &renderingUBOBuffersMapped[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 20;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 20;

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 20;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 20;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createGraphicsDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphicsDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        graphicsDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, graphicsDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

            VkDescriptorBufferInfo renderingUBOBufferInfo{};
            renderingUBOBufferInfo.buffer = renderingUBOBuffers[i];
            renderingUBOBufferInfo.offset = 0;
            renderingUBOBufferInfo.range = sizeof(RenderingUBO);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = skyboxImageView;
            imageInfo.sampler = skyboxSampler;

            VkDescriptorBufferInfo lagrangianPointsBufferInfo{};
            lagrangianPointsBufferInfo.buffer = lagrangianPointsBuffers[i];
            lagrangianPointsBufferInfo.offset = 0;
            lagrangianPointsBufferInfo.range = sizeof(LagrangianPoint) * lagrangianPointCount;

            VkDescriptorBufferInfo velocityBufferInfo{};
            velocityBufferInfo.buffer = velocityBuffers[i];
            velocityBufferInfo.offset = 0;
            velocityBufferInfo.range = Nxyz * sizeof(float) * 3;

            VkDescriptorBufferInfo flagsBufferInfo{};
            flagsBufferInfo.buffer = flagBuffers[i];
            flagsBufferInfo.offset = 0;
            flagsBufferInfo.range = Nxyz * sizeof(uint32_t);

            std::array<VkWriteDescriptorSet, 6> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &renderingUBOBufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &lagrangianPointsBufferInfo;

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &velocityBufferInfo;

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &flagsBufferInfo;

            
            VkDescriptorBufferInfo rigidBodyStateBufferInfo{};
            rigidBodyStateBufferInfo.buffer = rigidBodyStateBuffers[i];
            rigidBodyStateBufferInfo.offset = 0;
            rigidBodyStateBufferInfo.range = sizeof(RigidBodyState) * rigidBodyCount;
            
            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = graphicsDescriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pBufferInfo = &rigidBodyStateBufferInfo;
            
            vkUpdateDescriptorSets(device, 6, descriptorWrites.data(), 0, nullptr);
        }
    }

    void createComputeDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = sizeof(SimulateUBO);

            std::array<VkWriteDescriptorSet, 18> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            storageBufferInfoLastFrame.buffer = shaderStorageBuffers[i];
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range = sizeof(Particle) * particle_count;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

            VkDescriptorBufferInfo velocityBufferInfo{};
            velocityBufferInfo.buffer = velocityBuffers[i];
            velocityBufferInfo.offset = 0;
            velocityBufferInfo.range = Nxyz * sizeof(float) * 3;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = computeDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &velocityBufferInfo;

            VkDescriptorBufferInfo rhoBufferInfo{};
            rhoBufferInfo.buffer = rhoBuffers[i];
            rhoBufferInfo.offset = 0;
            rhoBufferInfo.range = Nxyz * sizeof(float);

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = computeDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &rhoBufferInfo;

            VkDescriptorBufferInfo flagBufferInfo{};
            flagBufferInfo.buffer = flagBuffers[i];
            flagBufferInfo.offset = 0;
            flagBufferInfo.range = Nxyz * sizeof(uint32_t);

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = computeDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &flagBufferInfo;

            VkDescriptorBufferInfo DDFBufferInfo{};
            DDFBufferInfo.buffer = DDFBuffers[i];
            DDFBufferInfo.offset = 0;
            DDFBufferInfo.range = Nxyz * sizeof(float) * Q;

            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = computeDescriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pBufferInfo = &DDFBufferInfo;

            VkDescriptorBufferInfo borderForceBufferInfo{};
            borderForceBufferInfo.buffer = borderForceBuffers[i];
            borderForceBufferInfo.offset = 0;
            borderForceBufferInfo.range = Nxyz * sizeof(float) * 3;

            descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[6].dstSet = computeDescriptorSets[i];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pBufferInfo = &borderForceBufferInfo;

            VkDescriptorBufferInfo lagrangianPointsBufferInfo{};
            lagrangianPointsBufferInfo.buffer = lagrangianPointsBuffers[i];
            lagrangianPointsBufferInfo.offset = 0;
            lagrangianPointsBufferInfo.range = sizeof(LagrangianPoint) * lagrangianPointCount;

            descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[7].dstSet = computeDescriptorSets[i];
            descriptorWrites[7].dstBinding = 7;
            descriptorWrites[7].dstArrayElement = 0;
            descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[7].descriptorCount = 1;
            descriptorWrites[7].pBufferInfo = &lagrangianPointsBufferInfo;

            VkDescriptorBufferInfo lagrangianDataBufferInfo{};
            lagrangianDataBufferInfo.buffer = lagrangianDataBuffers[i];
            lagrangianDataBufferInfo.offset = 0;
            lagrangianDataBufferInfo.range = sizeof(LagrangianData) * lagrangianPointCount;

            descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[8].dstSet = computeDescriptorSets[i];
            descriptorWrites[8].dstBinding = 8;
            descriptorWrites[8].dstArrayElement = 0;
            descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[8].descriptorCount = 1;
            descriptorWrites[8].pBufferInfo = &lagrangianDataBufferInfo;

            VkDescriptorBufferInfo lagrangianPointsRestBufferInfo{};
            lagrangianPointsRestBufferInfo.buffer = lagrangianPointsRestBuffers[i];
            lagrangianPointsRestBufferInfo.offset = 0;
            lagrangianPointsRestBufferInfo.range = sizeof(LagrangianPoint) * lagrangianPointCount;

            descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[9].dstSet = computeDescriptorSets[i];
            descriptorWrites[9].dstBinding = 9;
            descriptorWrites[9].dstArrayElement = 0;
            descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[9].descriptorCount = 1;
            descriptorWrites[9].pBufferInfo = &lagrangianPointsRestBufferInfo;

            VkDescriptorBufferInfo lagrangianPointsPrevBufferInfo{};
            lagrangianPointsPrevBufferInfo.buffer = lagrangianPointsPrevBuffers[i];
            lagrangianPointsPrevBufferInfo.offset = 0;
            lagrangianPointsPrevBufferInfo.range = sizeof(LagrangianPoint) * lagrangianPointCount;

            descriptorWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[10].dstSet = computeDescriptorSets[i];
            descriptorWrites[10].dstBinding = 10;
            descriptorWrites[10].dstArrayElement = 0;
            descriptorWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[10].descriptorCount = 1;
            descriptorWrites[10].pBufferInfo = &lagrangianPointsPrevBufferInfo;

            VkDescriptorBufferInfo tempForcesBufferInfo{};
            tempForcesBufferInfo.buffer = tempForcesBuffers[i];
            tempForcesBufferInfo.offset = 0;
            tempForcesBufferInfo.range = sizeof(glm::vec4) * lagrangianPointCount;

            descriptorWrites[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[11].dstSet = computeDescriptorSets[i];
            descriptorWrites[11].dstBinding = 11;
            descriptorWrites[11].dstArrayElement = 0;
            descriptorWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[11].descriptorCount = 1;
            descriptorWrites[11].pBufferInfo = &tempForcesBufferInfo;

            VkDescriptorBufferInfo totalForceTorqueBufferInfo{};
            totalForceTorqueBufferInfo.buffer = totalForceTorqueBuffers[i];
            totalForceTorqueBufferInfo.offset = 0;
            totalForceTorqueBufferInfo.range = sizeof(glm::vec4) * 2 * rigidBodyCount;

            descriptorWrites[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[12].dstSet = computeDescriptorSets[i];
            descriptorWrites[12].dstBinding = 12;
            descriptorWrites[12].dstArrayElement = 0;
            descriptorWrites[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[12].descriptorCount = 1;
            descriptorWrites[12].pBufferInfo = &totalForceTorqueBufferInfo;

            VkDescriptorBufferInfo rigidBodyStateBufferInfo{};
            rigidBodyStateBufferInfo.buffer = rigidBodyStateBuffers[i];
            rigidBodyStateBufferInfo.offset = 0;
            rigidBodyStateBufferInfo.range = sizeof(RigidBodyState) * rigidBodyCount;

            descriptorWrites[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[13].dstSet = computeDescriptorSets[i];
            descriptorWrites[13].dstBinding = 13;
            descriptorWrites[13].dstArrayElement = 0;
            descriptorWrites[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[13].descriptorCount = 1;
            descriptorWrites[13].pBufferInfo = &rigidBodyStateBufferInfo;

            VkDescriptorBufferInfo skBufferInfo{};
            skBufferInfo.buffer = skBuffers[i];
            skBufferInfo.offset = 0;
            skBufferInfo.range = sizeof(float) * lagrangianPointCount;

            descriptorWrites[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[14].dstSet = computeDescriptorSets[i];
            descriptorWrites[14].dstBinding = 14;
            descriptorWrites[14].dstArrayElement = 0;
            descriptorWrites[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[14].descriptorCount = 1;
            descriptorWrites[14].pBufferInfo = &skBufferInfo;

            VkDescriptorBufferInfo rigidBodyInfoBufferInfo{};
            rigidBodyInfoBufferInfo.buffer = rigidBodyInfoBuffers[i];
            rigidBodyInfoBufferInfo.offset = 0;
            rigidBodyInfoBufferInfo.range = sizeof(RigidBodyInfo) * rigidBodyCount;

            descriptorWrites[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[15].dstSet = computeDescriptorSets[i];
            descriptorWrites[15].dstBinding = 15;
            descriptorWrites[15].dstArrayElement = 0;
            descriptorWrites[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[15].descriptorCount = 1;
            descriptorWrites[15].pBufferInfo = &rigidBodyInfoBufferInfo;

            VkDescriptorBufferInfo bodyIndexBufferInfo{};
            bodyIndexBufferInfo.buffer = bodyIndexBuffers[i];
            bodyIndexBufferInfo.offset = 0;
            bodyIndexBufferInfo.range = sizeof(uint32_t) * lagrangianPointCount;

            descriptorWrites[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[16].dstSet = computeDescriptorSets[i];
            descriptorWrites[16].dstBinding = 16;
            descriptorWrites[16].dstArrayElement = 0;
            descriptorWrites[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[16].descriptorCount = 1;
            descriptorWrites[16].pBufferInfo = &bodyIndexBufferInfo;

            VkDescriptorBufferInfo tempVelBufferInfo{};
            tempVelBufferInfo.buffer = tempVelBuffers[i];
            tempVelBufferInfo.offset = 0;
            tempVelBufferInfo.range = Nxyz * sizeof(float) * 3;

            descriptorWrites[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[17].dstSet = computeDescriptorSets[i];
            descriptorWrites[17].dstBinding = 17;
            descriptorWrites[17].dstArrayElement = 0;
            descriptorWrites[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[17].descriptorCount = 1;
            descriptorWrites[17].pBufferInfo = &tempVelBufferInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createComputeCommandBuffers() {
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)computeCommandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearColors;
        clearColors[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearColors[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearColors.size());
        renderPassInfo.pClearValues = clearColors.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            
            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &skyboxBuffers[currentFrame], offsets);

            vkCmdBindIndexBuffer(commandBuffer, skyboxIndexBuffers[currentFrame], 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(skyboxIndices.size()), 1, 0, 0, 0);
        }

        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &wireframeBuffers[currentFrame], offsets);

            vkCmdBindIndexBuffer(commandBuffer, wireframeIndexBuffers[currentFrame], 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(wireframeIndices.size()), 1, 0, 0, 0);
        }

        {
            if (render_mode == 2) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, velocityPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

                VkViewport viewport{};
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.width = (float)swapChainExtent.width;
                viewport.height = (float)swapChainExtent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset = { 0, 0 };
                scissor.extent = swapChainExtent;
                vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                constexpr uint STRIDE = 6u;
                vkCmdDraw(commandBuffer, (Nx / STRIDE) * (Ny / STRIDE) * (Nz / STRIDE) * 2, 1, 0, 0);
            } else if (render_mode == 0 || render_mode == 1) {
                VkPipeline currentPipeline = (render_mode == 1) ? diagnosticPipeline : graphicsPipeline;
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, currentPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

                VkViewport viewport{};
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.width = (float)swapChainExtent.width;
                viewport.height = (float)swapChainExtent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset = { 0, 0 };
                scissor.extent = swapChainExtent;
                vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);

                vkCmdDraw(commandBuffer, particle_count, 1, 0, 0);
            }
        }

        // Model particle rendering (model_render_mode = 0)
        if (model_render_mode == 0) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lagrangianPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdDraw(commandBuffer, lagrangianPointCount, 1, 0, 0);
        }

        // Model mesh rendering (model_render_mode = 1)
        {
            bool shouldRenderMesh = (model_render_mode == 1) && !meshVertexBuffers.empty() && !meshIndexBuffers.empty();
            
            if (shouldRenderMesh) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

                VkViewport viewport{};
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.width = (float)swapChainExtent.width;
                viewport.height = (float)swapChainExtent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset = { 0, 0 };
                scissor.extent = swapChainExtent;
                vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                for (uint32_t bodyIdx = 0; bodyIdx < rigidBodyCount; ++bodyIdx) {
                    if (!meshVerticesPerBody[bodyIdx].empty() && !meshIndicesPerBody[bodyIdx].empty()) {
                        size_t vertexBufferIdx = bodyIdx * MAX_FRAMES_IN_FLIGHT + currentFrame;
                        size_t indexBufferIdx = bodyIdx * MAX_FRAMES_IN_FLIGHT + currentFrame;
                        
                        VkDeviceSize offsets[] = { 0 };
                        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &meshVertexBuffers[vertexBufferIdx], offsets);
                        vkCmdBindIndexBuffer(commandBuffer, meshIndexBuffers[indexBufferIdx], 0, VK_INDEX_TYPE_UINT32);
                        
                        vkCmdPushConstants(commandBuffer, graphicsPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t), &bodyIdx);
                        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(meshIndicesPerBody[bodyIdx].size()), 1, 0, 0, 0);
                    }
                }
            }
        }

        // Lagrangian Points rendering (IBM boundary points, only when model particle mode is active)
        if (enIBM && model_render_mode == 0) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lagrangianPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdDraw(commandBuffer, lagrangianPointCount, 1, 0, 0);
        }

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer, uint currentFrame) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VkMemoryBarrier memorybarrier{};
        memorybarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memorybarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memorybarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording compute command buffer!");
        }

        if (!isInit) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, initPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, Nxyz / 256 + 1, 1, 1);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

            isInit = true;
        }

        vkCmdFillBuffer(commandBuffer, borderForceBuffers[currentFrame], 0, Nxyz * sizeof(float) * 3, 0);
        VkMemoryBarrier fillmemorybarrier{};
        fillmemorybarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillmemorybarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillmemorybarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillmemorybarrier, 0, nullptr, 0, nullptr);

        if (enIBM) {
            {
                std::vector<RigidBodyInfo> updatedInfos;
                for (size_t i = 0; i < rigidBodyCount; i++) {
                    updatedInfos.push_back(rigidBodies[i].getRigidBodyInfo());
                }
                vkCmdUpdateBuffer(
                    commandBuffer,
                    rigidBodyInfoBuffers[currentFrame],
                    0,
                    sizeof(RigidBodyInfo) * rigidBodyCount,
                    updatedInfos.data());
                
                VkMemoryBarrier infoUpdateBarrier{};
                infoUpdateBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                infoUpdateBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                infoUpdateBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &infoUpdateBarrier,
                    0, nullptr,
                    0, nullptr);
            }

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, rigidBodySolverPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, rigidBodyCount, 1, 1);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, updatePositionsPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, lagrangianPointCount / 256 + 1, 1, 1);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, calcUPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer, Nxyz / 256 + 1, 1, 1);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

        if (enIBM) {
            {
                VkMemoryBarrier preCopyBarrier{};
                preCopyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                preCopyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                preCopyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &preCopyBarrier, 0, nullptr, 0, nullptr);
            }
            
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = Nxyz * sizeof(float) * 3;
            vkCmdCopyBuffer(commandBuffer, velocityBuffers[currentFrame], tempVelBuffers[currentFrame], 1, &copyRegion);

            VkMemoryBarrier copyBarrier{};
            copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copyBarrier, 0, nullptr, 0, nullptr);

            for (int i = 0; i < 5; ++i) {
                int iteration = i;

                {
                    VkMemoryBarrier preUpdateBarrier{};
                    preUpdateBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    preUpdateBarrier.srcAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
                    preUpdateBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &preUpdateBarrier, 0, nullptr, 0, nullptr);
                }

                vkCmdUpdateBuffer(
                    commandBuffer,
                    uniformBuffers[currentFrame],
                    offsetof(SimulateUBO, fixed_point_iteration),
                    sizeof(int),
                    &iteration);

                VkMemoryBarrier uboUpdateBarrier{};
                uboUpdateBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                uboUpdateBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                uboUpdateBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;

                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &uboUpdateBarrier,
                    0, nullptr,
                    0, nullptr);
                
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ibmForce1Pipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
                vkCmdDispatch(commandBuffer, lagrangianPointCount / 256 + 1, 1, 1);
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
                
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ibmForce2Pipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
                vkCmdDispatch(commandBuffer, lagrangianPointCount / 256 + 1, 1, 1);
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
            }

            vkCmdFillBuffer(commandBuffer, totalForceTorqueBuffers[currentFrame], 0, sizeof(glm::vec4) * 2 * rigidBodyCount, 0);
            VkMemoryBarrier fillForceTorqueBarrier{};
            fillForceTorqueBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            fillForceTorqueBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fillForceTorqueBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillForceTorqueBarrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, forceReductionPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, lagrangianPointCount / 256 + 1, 1, 1);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collideAndStreamPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer, Nxyz / 256 + 1, 1, 1);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

        for (uint32_t iter = 0; iter < 1; iter++) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, particle_count / 256 + 1, 1, 1);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        }

        if (enIBM) {
            {
                VkMemoryBarrier preLagCopyBarrier{};
                preLagCopyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                preLagCopyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                preLagCopyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &preLagCopyBarrier, 0, nullptr, 0, nullptr);
            }
            
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = sizeof(LagrangianPoint) * lagrangianPointCount;
            vkCmdCopyBuffer(commandBuffer, lagrangianPointsBuffers[currentFrame], lagrangianPointsPrevBuffers[currentFrame], 1, &copyRegion);

            VkMemoryBarrier copyBarrier{};
            copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copyBarrier, 0, nullptr, 0, nullptr);
        }


        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }

    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(swapChainImages.size());
        computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render finished semaphore!");
            }
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics synchronization objects for a frame!");
            }
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute synchronization objects for a frame!");
            }
        }
    }

    void updateUniformBuffer(uint32_t currentFrame) {
        {
            glm::vec3 cameraPos = glm::vec3(1.0f);
            cameraPos.x = distance * sin(glm::radians(ry)) * cos(glm::radians(rx));
            cameraPos.y = distance * sin(glm::radians(ry)) * sin(glm::radians(rx));
            cameraPos.z = distance * cos(glm::radians(ry));
            RenderingUBO ubo{};
            ubo.Nx = Nx;
            ubo.Ny = Ny;
            ubo.Nz = Nz;
            ubo.render_mode = render_mode;
            ubo.model = glm::mat4(1.0f);
            ubo.view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
            ubo.proj[1][1] *= -1;
            memcpy(renderingUBOBuffersMapped[currentFrame], &ubo, sizeof(ubo));
        }

        {
            SimulateUBO ubo{};
            ubo.Nx = Nx;
            ubo.Ny = Ny;
            ubo.Nz = Nz;
            ubo.Nxyz = Nxyz;
            ubo.particleCount = particle_count;
            ubo.particleRho = 1.0f;
            ubo.niu = 0.01f;
            ubo.tau = 3.0f * ubo.niu + 0.5f;
            ubo.inv_tau = 1.0f / ubo.tau;
            ubo.fx = 0.0f;
            ubo.fy = 0.0f;
            ubo.fz = 0.0f;
            ubo.dt = lastFrameTime / 1000.0f;
            ubo.t = currentTime;
            ubo.render_mode = render_mode;
            ubo.lagrangianPointCount = lagrangianPointCount;
            ubo.couplingStrength = couplingStrength;
            ubo.rigidBodyCount = rigidBodyCount;
            ubo.useEmitter = useEmitter;
            ubo.spawnRate = spawnRate;
            ubo.emitterPos = glm::vec4(emitterPos[0], emitterPos[1], emitterPos[2], emitterPos[3]);
            ubo.emitterVel = glm::vec4(emitterVel[0], emitterVel[1], emitterVel[2], emitterVel[3]);

            memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));

            currentTime += 1;
        }
    }

    void drawFrame() {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // Compute submission        
        vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        updateUniformBuffer(currentFrame);

        vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

        vkResetCommandBuffer(computeCommandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordComputeCommandBuffer(computeCommandBuffers[currentFrame], currentFrame);

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        };

        // Graphics submission
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSemaphore waitSemaphores[] = { computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphores[imageIndex];

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        //barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        //barrier.subresourceRange.baseMipLevel = 0;
        //barrier.subresourceRange.levelCount = 1;
        //barrier.subresourceRange.baseArrayLayer = 0;
        //barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange = subresourceRange;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                indices.graphicsAndComputeFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        //extensions.push_back("VK_KHR_shader_buffer_float32_atomic_add");
        extensions.push_back("VK_KHR_get_physical_device_properties2");

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    ComputeShaderApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#version 450
#extension GL_EXT_scalar_block_layout : require

layout (binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    uint render_mode;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

struct RigidBodyState {
    vec4 position;
    vec4 orientation;
    vec4 lin_vel;
    vec4 ang_vel;
    mat4 modelMatrix;
};

layout(std430, binding = 5) readonly buffer RigidBodyStateBuffer {
    RigidBodyState bodies[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;

layout(push_constant) uniform PushConstants {
    uint bodyIndex;
} pc;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - ubo.Nx / 2.0f) / factor,
                (pos.y - ubo.Ny / 2.0f) / factor,
                (pos.z - ubo.Nz / 2.0f) / factor);
}

void main() {
    mat4 modelMatrix = bodies[pc.bodyIndex].modelMatrix;
    vec4 worldPos = modelMatrix * vec4(inPosition, 1.0);
    
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    vec3 normalizedPos = particle_norm(worldPos.xyz, factor);
    gl_Position = ubo.proj * ubo.view * vec4(normalizedPos, 1.0f);
    
    fragWorldPos = normalizedPos;
    
    mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
    fragNormal = normalize(normalMatrix * -inNormal);
}

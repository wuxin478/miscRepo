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

struct LagrangianPoint {
    vec4 position;
};

layout(std430, binding = 2) readonly buffer LagrangianPoints {
    LagrangianPoint lagrangianPoints[];
};

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - ubo.Nx / 2.0f) / factor,
                (pos.y - ubo.Ny / 2.0f) / factor,
                (pos.z - ubo.Nz / 2.0f) / factor);
}

void main() {
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    vec3 result = particle_norm(lagrangianPoints[gl_VertexIndex].position.xyz, factor);
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(result, 1.0);
    gl_PointSize = 4.0;
}

#version 450
#extension GL_EXT_scalar_block_layout : require

#define Q 9

layout (binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    uint render_mode;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragAlpha;
layout(location = 2) flat out uint fragRenderMode;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - ubo.Nx / 2.0f) / factor, (pos.y - ubo.Ny / 2.0f) / factor, (pos.z - ubo.Nz / 2.0f) / factor);
}

void main() {
    if (inColor.a <= 0.0) {
        gl_Position = vec4(1e10, 1e10, 1e10, 1.0);
        return;
    }
    
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(particle_norm(inPosition.xyz, factor), 1.0);
    
    fragRenderMode = ubo.render_mode;
    
    if (ubo.render_mode == 1) {
        gl_PointSize = 2.0;
        fragColor = vec3(0.2, 1.0, 0.2);
        fragAlpha = 1.0;
    } else {
        float randSize = 0.8 + fract(sin(float(gl_VertexIndex)) * 43758.5453) * 0.4;
        gl_PointSize = 15.0 * randSize;
        
        fragColor = inColor.rgb;
        fragAlpha = 1.0;
    }
}

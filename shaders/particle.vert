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

layout(location = 0) in vec4 inPosition;  // xyz: position, w: age (current lifetime in seconds)
layout(location = 1) in vec4 inColor;      // rgb: color, a: life (total lifetime in seconds)

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragAlpha;
layout(location = 2) flat out uint fragRenderMode;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - ubo.Nx / 2.0f) / factor, (pos.y - ubo.Ny / 2.0f) / factor, (pos.z - ubo.Nz / 2.0f) / factor);
}

void main() {
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(particle_norm(inPosition.xyz, factor), 1.0);
    
    fragRenderMode = ubo.render_mode;
    
    if (ubo.render_mode == 1) {
        gl_PointSize = 2.0;
        fragColor = vec3(0.2, 1.0, 0.2);
        fragAlpha = 1.0;
    } else {
        float progress = inPosition.w / inColor.a;
        float randSize = 0.8 + fract(sin(float(gl_VertexIndex)) * 43758.5453) * 0.4;
        gl_PointSize = 15.0 * (1.0 + progress * 1.5) * randSize;
        
        float alpha = clamp(1.0 - progress, 0.0, 1.0);
        
        vec3 startColor = vec3(0.9, 0.8, 0.2);
        vec3 endColor = vec3(0.5, 0.5, 0.5);
        vec3 finalColor = mix(startColor, endColor, progress);
        
        fragColor = finalColor;
        fragAlpha = alpha;
    }
}

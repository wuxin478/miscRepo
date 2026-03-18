#version 450
#extension GL_EXT_scalar_block_layout : require

layout(std140, binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    uint render_mode;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(std430, binding = 3) readonly buffer Velocity {
    float vels[];
};

layout(std430, binding = 4) readonly buffer Flags {
    uint flags[];
};

layout(location = 0) out vec3 vColor;

const uint STRIDE = 6u;

vec3 loadVelocity(uint n) {
    uint Nxyz = ubo.Nx * ubo.Ny * ubo.Nz;
    return vec3(vels[n], vels[n + Nxyz], vels[n + 2u * Nxyz]);
}

vec3 velocityToColor(float speed, float maxSpeed) {
    float t = clamp(speed / maxSpeed, 0.0, 1.0);
    
    vec3 lowColor = vec3(0.0, 0.0, 1.0);
    vec3 midColor = vec3(0.0, 1.0, 0.0);
    vec3 highColor = vec3(1.0, 0.0, 0.0);
    
    vec3 color;
    if (t < 0.5) {
        color = mix(lowColor, midColor, t * 2.0);
    } else {
        color = mix(midColor, highColor, (t - 0.5) * 2.0);
    }
    
    return color;
}

void main() {
    uint lineIndex = gl_VertexIndex / 2;
    uint endpoint = gl_VertexIndex % 2;
    
    uint strideNx = ubo.Nx / STRIDE;
    uint strideNy = ubo.Ny / STRIDE;
    uint strideNz = ubo.Nz / STRIDE;
    uint strideNxy = strideNx * strideNy;
    
    uint sz = lineIndex / strideNxy;
    uint sremainder = lineIndex - sz * strideNxy;
    uint sy = sremainder / strideNx;
    uint sx = sremainder - sy * strideNx;
    
    uint x = sx * STRIDE;
    uint y = sy * STRIDE;
    uint z = sz * STRIDE;
    
    uint Nxy = ubo.Nx * ubo.Ny;
    uint originalIndex = z * Nxy + y * ubo.Nx + x;
    
    float factor = max(ubo.Nx, max(ubo.Ny, ubo.Nz));
    
    vec3 vel = loadVelocity(originalIndex);
    float speed = length(vel);
    
    vec3 pos = vec3(float(x), float(y), float(z));
    
    if (endpoint == 1 && speed > 0.0001) {
        pos = pos + normalize(vel) * speed * 250.0f;
    }
    
    vec3 normalizedPos = vec3(
        (pos.x - ubo.Nx / 2.0) / factor,
        (pos.y - ubo.Ny / 2.0) / factor,
        (pos.z - ubo.Nz / 2.0) / factor
    );
    
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(normalizedPos, 1.0);
    
    float maxSpeed = 0.15;
    vColor = velocityToColor(speed, maxSpeed);
    
    uint flag = flags[originalIndex];
    const uint TYPE_S = 1u;
    if (flag == TYPE_S) {
        vColor = vec3(0.5, 0.5, 0.5);
    }
}

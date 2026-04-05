#ifndef SIMULATE_HEADER_GLSL
#define SIMULATE_HEADER_GLSL

#define D 3
#define Q 19 
#define SCALE 1e10
#define def_c 0.57735027f

#define def_w0 (1.0f/3.0f)
#define def_ws (1.0f/18.0f)
#define def_we (1.0f/36.0f)

#define TYPE_S  0x01
#define TYPE_E  0x02
#define TYPE_T  0x04
#define TYPE_F  0x08
#define TYPE_I  0x10
#define TYPE_G  0x20
#define TYPE_X  0x40
#define TYPE_Y  0x80

#define TYPE_MS 0x03
#define TYPE_BO 0x03
#define TYPE_IF 0x18
#define TYPE_IG 0x30
#define TYPE_GI 0x38
#define TYPE_SU 0x38

const float w[Q] = float[](
    1.0f /  3.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 18.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f,
    1.0f / 36.0f
);

const ivec3 e[Q] = ivec3[](
    ivec3( 0,  0,  0),
    ivec3( 1,  0,  0),
    ivec3(-1,  0,  0),
    ivec3( 0,  1,  0),
    ivec3( 0, -1,  0),
    ivec3( 0,  0,  1),
    ivec3( 0,  0, -1),
    ivec3( 1,  1,  0),
    ivec3(-1, -1,  0),
    ivec3( 1,  0,  1),
    ivec3(-1,  0, -1),
    ivec3( 0,  1,  1),
    ivec3( 0, -1, -1),
    ivec3( 1, -1,  0),
    ivec3(-1,  1,  0),
    ivec3( 1,  0, -1),
    ivec3(-1,  0,  1),
    ivec3( 0,  1, -1),
    ivec3( 0, -1,  1)
);

struct Particle {
    vec4 position;
    vec4 color;
};

struct LagrangianPoint {
    vec4 position;
};

struct LagrangianData {
    vec4 velocity;
    vec4 force;
};

struct RigidBodyState {
    vec4 position;
    vec4 orientation;
    vec4 lin_vel;
    vec4 ang_vel;
    mat4 modelMatrix;
};

struct RigidBodyInfo {
    float mass;
    float inv_mass;
    mat4 invInertiaTensor;
    float volume;
    uint manualMode;
    vec4 manualLinVel;
    vec4 manualAngVel;
};

struct TotalForceTorque {
    vec4 total_force;
    vec4 total_torque;
};

layout(std430, binding = 0) uniform SimulateUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    uint Nxyz;
    uint particleCount;
    float particleRho;
    float niu;
    float tau;
    float inv_tau;
    float fx;
    float fy;
    float fz;
    float dt;
    uint t;
    uint render_mode;
    uint fixed_point_iteration;
    uint lagrangianPointCount;
    float couplingStrength;
    uint rigidBodyCount;
    uint useEmitter;
    float spawnRate;
    vec4 emitterPos;
    vec4 emitterVel;
} ubo;

layout(std430, binding = 1) buffer ParticlesBuffer {
    Particle particles[];
};

layout(std430, binding = 2) buffer VelocityBuffer {
    float vels[];
};

layout(std430, binding = 3) buffer RhoBuffer {
    float rhos[];
};

layout(std430, binding = 4) buffer FlagBuffer {
    uint flags[];
};

layout(std430, binding = 5) buffer DDFBuffer {
    float ddfs[];
};

layout(std430, binding = 6) buffer BorderForceBuffer {
    float bfs[];
};

layout(std430, binding = 7) buffer LagrangianPointsBuffer {
    LagrangianPoint lagrangianPoints[];
};

layout(std430, binding = 8) buffer LagrangianDataBuffer {
    LagrangianData lagrangianData[];
};

layout(std430, binding = 9) buffer LagrangianPointsRestBuffer {
    LagrangianPoint lagrangianPoints_Rest[];
};

layout(std430, binding = 10) buffer LagrangianPointsPrevBuffer {
    LagrangianPoint lagrangianPoints_Prev[];
};

layout(std430, binding = 11) buffer TempForceBuffer {
    vec4 tempForces[];
};

layout(std430, binding = 12) buffer TotalForceTorqueBuffer {
    vec4 total_force_torque[];
};

layout(std430, binding = 13) buffer RigidBodyStateBuffer {
    RigidBodyState bodies[];
};

layout(std430, binding = 14) buffer SkBuffer {
    float s_k_values[];
};

layout(std430, binding = 15) buffer RigidBodyInfoBuffer {
    RigidBodyInfo bodyInfos[];
};

layout(std430, binding = 16) buffer BodyIndexBuffer {
    uint bodyIndices[];
};

layout(std430, binding = 17) buffer TempVelocityBuffer {
    float tempVels[];
};

uint index_f(uint n, uint i) {
    return i * ubo.Nxyz + n;
}

int c(const uint i) {
    return e[i % Q][i / Q];
}

float sq(float x) {
    return x * x;
}

uvec3 coordinates(const uint n) {
    const uint t = uint(n % (ubo.Nx * ubo.Ny));
    return uvec3(t % ubo.Nx, t / ubo.Nx, uint(n / (ubo.Nx * ubo.Ny)));
}

void calculate_indices(uint n, inout uint x0, inout uint xp, inout uint xm,
                       inout uint y0, inout uint yp, inout uint ym,
                       inout uint z0, inout uint zp, inout uint zm) {
    const uvec3 xyz = coordinates(n);
    x0 = uint(xyz.x);
    xp = uint((xyz.x + 1u) % ubo.Nx);
    xm = uint((xyz.x + ubo.Nx - 1u) % ubo.Nx);
    y0 = uint(xyz.y * ubo.Nx);
    yp = uint(((xyz.y + 1u) % ubo.Ny) * ubo.Nx);
    ym = uint(((xyz.y + ubo.Ny - 1u) % ubo.Ny) * ubo.Nx);
    z0 = uint(xyz.z * uint(ubo.Ny * ubo.Nx));
    zp = uint((xyz.z + 1u) % ubo.Nz) * uint(ubo.Ny * ubo.Nx);
    zm = uint((xyz.z + ubo.Nz - 1u) % ubo.Nz) * uint(ubo.Ny * ubo.Nx);
}

void neighbors(uint index, inout uint nbs[Q]) {
    uint x0, xp, xm, y0, yp, ym, z0, zp, zm;
    calculate_indices(index, x0, xp, xm, y0, yp, ym, z0, zp, zm);
    nbs[ 0] = index;
    nbs[ 1] = xp + y0 + z0; nbs[ 2] = xm + y0 + z0;
    nbs[ 3] = x0 + yp + z0; nbs[ 4] = x0 + ym + z0;
    nbs[ 5] = x0 + y0 + zp; nbs[ 6] = x0 + y0 + zm;
    nbs[ 7] = xp + yp + z0; nbs[ 8] = xm + ym + z0;
    nbs[ 9] = xp + y0 + zp; nbs[10] = xm + y0 + zm;
    nbs[11] = x0 + yp + zp; nbs[12] = x0 + ym + zm;
    nbs[13] = xp + ym + z0; nbs[14] = xm + yp + z0;
    nbs[15] = xp + y0 + zm; nbs[16] = xm + y0 + zp;
    nbs[17] = x0 + yp + zm; nbs[18] = x0 + ym + zp;
}

void load_f(uint n, inout float fhn[Q], uint nbs[Q], uint t) {
    fhn[0] = ddfs[index_f(n, 0u)];
    for (uint q = 1u; q < Q; q += 2u) {
        fhn[q]   = ddfs[index_f(n,      t % 2u != 0u ? q     : q + 1u)];
        fhn[q+1u] = ddfs[index_f(nbs[q], t % 2u != 0u ? q + 1u : q)];
    }
}

void store_f(uint n, inout float fhn[Q], uint nbs[Q], uint t) {
    ddfs[index_f(n, 0u)] = fhn[0];
    for (uint q = 1u; q < Q; q += 2u) {
        ddfs[index_f(n,      t % 2u != 0u ? q     : q + 1u)] = fhn[q + 1u];
        ddfs[index_f(nbs[q], t % 2u != 0u ? q + 1u : q)]     = fhn[q];
    }
}

void apply_moving_boundaries(inout float fhn[Q], const uint nbs[Q]) {
    for (uint q = 1u; q < Q; q += 2u) {
        float w6 = -6.0f * w[q];
        uint ji;
        ji = nbs[q + 1u];
        fhn[q]   = (flags[ji] & TYPE_BO) == TYPE_S
            ? w6 * (c(q + 1u) * vels[ji] + c(Q + q + 1u) * vels[ubo.Nxyz + ji] + c(2u * Q + q + 1u) * vels[2u * ubo.Nxyz + ji]) + fhn[q]
            : fhn[q];
        ji = nbs[q];
        fhn[q + 1u] = (flags[ji] & TYPE_BO) == TYPE_S
            ? w6 * (c(q) * vels[ji] + c(Q + q) * vels[ubo.Nxyz + ji] + c(2u * Q + q) * vels[2u * ubo.Nxyz + ji]) + fhn[q + 1u]
            : fhn[q + 1u];
    }
}

void calculate_rho_u(const float f[Q], inout float rhon, inout float uxn, inout float uyn, inout float uzn) {
    float rho = f[0];
    for (uint q = 1u; q < Q; q++) rho += f[q];
    rho += 1.0f;
    float ux = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16];
    float uy = f[3] - f[4] + f[7] - f[8] + f[11] - f[12] + f[14] - f[13] + f[17] - f[18];
    float uz = f[5] - f[6] + f[9] - f[10] + f[11] - f[12] + f[16] - f[15] + f[18] - f[17];
    rhon = rho;
    uxn = ux / rho;
    uyn = uy / rho;
    uzn = uz / rho;
}

void calculate_forcing_terms(const float ux, const float uy, const float uz,
                             const float fx, const float fy, const float fz,
                             inout float Fin[Q]) {
    const float uF = -0.33333334f * (ux * fx + uy * fy + uz * fz);
    Fin[0] = 9.0f * def_w0 * uF;
    for (uint q = 1u; q < Q; q++) {
        Fin[q] = 9.0f * w[q] * (
            (c(q) * fx + c(Q + q) * fy + c(2u * Q + q) * fz) *
            (c(q) * ux + c(Q + q) * uy + c(2u * Q + q) * uz + 0.33333334f) + uF
        );
    }
}

void calculate_f_eq(const float rho, float ux, float uy, float uz, inout float feq[Q]) {
    const float rhom1 = rho - 1.0f;
    const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz));
    ux *= 3.0f;
    uy *= 3.0f;
    uz *= 3.0f;
    feq[0] = w[0] * (rho * 0.5f * c3 + rhom1);
    const float u0 = ux + uy, u1 = ux + uz, u2 = uy + uz, u3 = ux - uy, u4 = ux - uz, u5 = uy - uz;
    const float rhos = def_ws * rho, rhoe = def_we * rho;
    const float rhom1s = def_ws * rhom1, rhom1e = def_we * rhom1;
    feq[ 1] = rhos * (0.5f * (ux * ux + c3) + ux) + rhom1s;
    feq[ 2] = rhos * (0.5f * (ux * ux + c3) - ux) + rhom1s;
    feq[ 3] = rhos * (0.5f * (uy * uy + c3) + uy) + rhom1s;
    feq[ 4] = rhos * (0.5f * (uy * uy + c3) - uy) + rhom1s;
    feq[ 5] = rhos * (0.5f * (uz * uz + c3) + uz) + rhom1s;
    feq[ 6] = rhos * (0.5f * (uz * uz + c3) - uz) + rhom1s;
    feq[ 7] = rhoe * (0.5f * (u0 * u0 + c3) + u0) + rhom1e;
    feq[ 8] = rhoe * (0.5f * (u0 * u0 + c3) - u0) + rhom1e;
    feq[ 9] = rhoe * (0.5f * (u1 * u1 + c3) + u1) + rhom1e;
    feq[10] = rhoe * (0.5f * (u1 * u1 + c3) - u1) + rhom1e;
    feq[11] = rhoe * (0.5f * (u2 * u2 + c3) + u2) + rhom1e;
    feq[12] = rhoe * (0.5f * (u2 * u2 + c3) - u2) + rhom1e;
    feq[13] = rhoe * (0.5f * (u3 * u3 + c3) + u3) + rhom1e;
    feq[14] = rhoe * (0.5f * (u3 * u3 + c3) - u3) + rhom1e;
    feq[15] = rhoe * (0.5f * (u4 * u4 + c3) + u4) + rhom1e;
    feq[16] = rhoe * (0.5f * (u4 * u4 + c3) - u4) + rhom1e;
    feq[17] = rhoe * (0.5f * (u5 * u5 + c3) + u5) + rhom1e;
    feq[18] = rhoe * (0.5f * (u5 * u5 + c3) - u5) + rhom1e;
}

void apply_inflow_boundary(inout float fhn[Q], uint index) {
    if ((flags[index] & TYPE_BO) == TYPE_E) {
        float u_in = vels[2u * ubo.Nxyz + index];
        if (u_in > 0.0f) {
            float feq[Q];
            calculate_f_eq(1.0f, 0.0f, 0.0f, u_in, feq);
            for (uint q = 0u; q < Q; q++) {
                fhn[q] = feq[q];
            }
        }
    }
}

void apply_outflow_boundary(inout float fhn[Q], const uint nbs[Q]) {
    for (uint q = 1u; q < Q; q += 2u) {
        if ((flags[nbs[q + 1u]] & TYPE_BO) == TYPE_E) {
            float u_nb = vels[2u * ubo.Nxyz + nbs[q + 1u]];
            if (u_nb <= 0.0f) {
                fhn[q] = fhn[q + 1u];
            }
        }
        if ((flags[nbs[q]] & TYPE_BO) == TYPE_E) {
            float u_nb = vels[2u * ubo.Nxyz + nbs[q]];
            if (u_nb <= 0.0f) {
                fhn[q + 1u] = fhn[q];
            }
        }
    }
}

float peskin_w(float r) {
    float abs_r = abs(r);
    if (abs_r <= 1.0) {
        return 0.125 * (3.0 - 2.0 * abs_r + sqrt(max(0.0, 1.0 + 4.0 * abs_r - 4.0 * abs_r * abs_r)));
    } else if (abs_r <= 2.0) {
        return 0.125 * (5.0 - 2.0 * abs_r - sqrt(max(0.0, -7.0 + 12.0 * abs_r - 4.0 * abs_r * abs_r)));
    }
    return 0.0;
}

float compute_weight_3d(vec3 lagrangian_pos, vec3 eulerian_pos) {
    vec3 r = lagrangian_pos - eulerian_pos;
    return peskin_w(r.x) * peskin_w(r.y) * peskin_w(r.z);
}

#endif

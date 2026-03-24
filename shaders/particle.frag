#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragAlpha;
layout(location = 2) flat in uint fragRenderMode;

layout(location = 0) out vec4 outColor;

void main() {
    if (fragRenderMode == 1) {
        float dist = length(gl_PointCoord - vec2(0.5));
        if (dist > 0.5) discard;
        outColor = vec4(fragColor, 1.0);
    } else {
        float dist = length(gl_PointCoord - vec2(0.5));
        float mask = smoothstep(0.5, 0.2, dist);
        float finalAlpha = mask * 0.5;
        outColor = vec4(fragColor, finalAlpha);
    }
}

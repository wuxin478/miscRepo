#version 450

layout(location = 0) out vec4 outColor;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}

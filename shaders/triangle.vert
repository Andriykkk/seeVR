#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
};

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    frag_color = in_color;
}

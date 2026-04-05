#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 frag_world_pos;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 dx = dFdx(frag_world_pos);
    vec3 dy = dFdy(frag_world_pos);
    vec3 normal = normalize(cross(dx, dy));

    vec3 light1 = normalize(vec3(1.0, 1.0, 0.5));
    vec3 light2 = normalize(vec3(-1.0, 0.8, -1.0));

    float diffuse = max(dot(normal, light1), 0.0) * 0.6
                  + max(dot(normal, light2), 0.0) * 0.3;
    float ambient = 0.3;

    vec3 lit = frag_color * (ambient + diffuse);
    out_color = vec4(lit, 1.0);
}

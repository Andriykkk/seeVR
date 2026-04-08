material_albedo:    [MAX_MATERIALS] float3  — base color
material_roughness: [MAX_MATERIALS] float   — 0=mirror, 1=diffuse
material_metallic:  [MAX_MATERIALS] float   — 0=dielectric, 1=metal
material_emission:  [MAX_MATERIALS] float3  — emissive 

[ ] remove triangles color from pathtrasing
[ ] check buffers, does they copied on gpu, also some not initialised on gpu but stay on cpu, like tri_count for geoms


Materials for path tracing
Albedo (base color) — the color of the surface. When a ray hits, the bounce carries this color as a multiplier. Red surface = red albedo = only red light bounces.

Roughness (0-1) — how scattered the reflection is:

0 = perfect mirror. Reflected ray = reflect(incoming, normal). Sharp reflection.
1 = fully diffuse. Reflected ray = random cosine-weighted hemisphere sample. Matte surface.
0.3 = glossy. Reflected ray = mix between mirror reflection and random, using GGX or similar microfacet distribution.
Metallic (0-1) — how the surface reflects:

0 = dielectric (plastic, wood, rubber). Reflects white light (specular is colorless). Albedo only tints diffuse bounces.
1 = metal (gold, copper, steel). Reflects tinted light — specular reflection is multiplied by albedo. No diffuse component. Gold reflects yellow because albedo=(1, 0.8, 0.3) tints the reflection.
Emission — the surface emits light. A lamp is emission=(10,10,10) with any albedo. When a ray hits an emissive surface, it adds throughput * emission to the pixel color. This is how area lights work in path tracing.

Glass / transparent materials
Glass needs two special rays:

Refraction — ray passes through the surface, bending by Snell's law:


sin(theta_out) = sin(theta_in) * n1/n2
Where n1/n2 is the ratio of refractive indices (air=1.0, glass=1.5, water=1.33, diamond=2.42).

Reflection — part of the light bounces off the surface (Fresnel effect). At shallow angles, more light reflects (that's why water looks mirror-like at grazing angles).

Fresnel decides the split — Schlick's approximation:


R0 = ((n1-n2)/(n1+n2))^2
reflectance = R0 + (1-R0) * (1 - cos_theta)^5
For glass: R0 ≈ 0.04 (4% reflection at head-on, ~100% at grazing).

Total internal reflection — when a ray inside glass hits the surface at a steep angle, sin(theta_out) > 1 → no refraction possible → 100% reflection. This is what makes diamonds sparkle.

Implementation in shader:


if (material is glass) {
    float ior = 1.5;
    float eta = entering ? (1.0/ior) : ior;  // air→glass or glass→air
    float cos_i = abs(dot(ray_dir, normal));
    float R0 = pow((1.0-ior)/(1.0+ior), 2.0);
    float fresnel = R0 + (1.0-R0) * pow(1.0-cos_i, 5.0);
    
    if (random() < fresnel || total_internal_reflection) {
        ray_dir = reflect(ray_dir, normal);  // reflection
    } else {
        ray_dir = refract(ray_dir, normal, eta);  // refraction
        // flip normal tracking (now inside glass)
    }
    // glass albedo tints transmitted light (colored glass)
}
What to add to material:

ior (index of refraction) — 0 means opaque, >0 means transparent
Or simpler: transparency (0-1) + ior
So the full material would be:


albedo:      float3  — base color
roughness:   float   — 0=mirror, 1=diffuse  
metallic:    float   — 0=dielectric, 1=metal
emission:    float3  — emissive light color
ior:         float   — index of refraction (0=opaque, 1.5=glass)
Want me to implement materials in the shader and data?
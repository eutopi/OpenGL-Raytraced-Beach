# OpenGL-Raytraced-Beach

The goal of this project is to construct a ray tracer program that renders a 3D beach scene using OpenGL.

<p align="center"><img src="https://github.com/eutopi/OpenGL-Raytraced-Beach/blob/master/raytracingbeach.png" alt="drawing" width="500"/></p>
<p align="center"><i></i></p>

## Features
1. **Beach** - a sandy dune constructed with a quadric using diffuse BRDF
2. **Parasol** - a clipped sphere on a clipped cylindrical pole, also shaded with diffuse BRDF
3. **Beach ball** - a sphere with simple procedural texturing and Phong-Blinn BRDF
4. **Shadows** - there is a directional light source that contributes to the illumination of all objects with non-ideal (Lambertian/Phong-Blinn) BRDFs. Objects cast shadows on one another
5. **Disco** - omnidirectional, isotropic point lights
6. **Ocean** - an infinite plane of ideally reflective water surrounding the dune
7. **Flotsam** -  cargo box washed upon the shore constructed by clipping an infinite slab quadric with two orthogonal infinite slab quadrics
8. **This side up** - the box has procedural wood texturing
9. **Glass** - an object that both reflects and refracts light
10. **Fresnel** - the reflectance and transmittance of ideally reflective/refractive surfaces are found using the approximate Fresnel formula

## Libraries
- OpenGL
- GLUT


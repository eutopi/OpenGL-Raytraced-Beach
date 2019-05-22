//
//  main.cpp
//  RayTracingBeach
//
//  Created by Tongyu Zhou on 5/17/19.
//  Copyright © 2019 AIT. All rights reserved.
//

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

#include <vector>

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat4x4.h"

const unsigned int windowWidth = 700, windowHeight = 700;

int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle)
{
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0)
    {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

void checkShader(unsigned int shader, char * message)
{
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK)
    {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

void checkLinking(unsigned int program)
{
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK)
    {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

class Shader
{
protected:
    unsigned int shaderProgram;
    
public:
    Shader()
    {
        const char *vertexSource = "\n\
        #version 410 \n\
        precision highp float; \n\
        \n\
        in vec2 vertexPosition;    \n\
        in vec2 vertexTexCoord; \n\
        out vec2 texCoord; \n\
        \n\
        void main() \n\
        { \n\
        texCoord = vertexTexCoord; \n\
        gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); \n\
        } \n\
        ";
        
        const char *fragmentSource = "\n\
        #version 410 \n\
        precision highp float; \n\
        \n\
        uniform sampler2D samplerUnit; \n\
        in vec2 texCoord;  \n\
        out vec4 fragmentColor; \n\
        \n\
        void main() { \n\
        fragmentColor = texture(samplerUnit, texCoord);  \n\
        } \n\
        ";
        
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
        
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");
        
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
        
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");
        
        shaderProgram = glCreateProgram();
        if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
        
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        
        glBindAttribLocation(shaderProgram, 0, "vertexPosition");
        glBindAttribLocation(shaderProgram, 1, "vertexTexCoord");
        
        glBindFragDataLocation(shaderProgram, 0, "fragmentColor");
        
        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
    }
    
    ~Shader()
    {
        if(shaderProgram) glDeleteProgram(shaderProgram);
    }
    
    void Run()
    {
        if(shaderProgram) glUseProgram(shaderProgram);
    }
    
    void UploadSamplerID()
    {
        int samplerUnit = 0;
        int location = glGetUniformLocation(shaderProgram, "samplerUnit");
        glUniform1i(location, samplerUnit);
        glActiveTexture(GL_TEXTURE0 + samplerUnit);
    }
};

Shader *shader = 0;

// Simple material class, with object color, and headlight shading.

class Material
{
protected:
    vec3 frontColor, backColor;
    vec3 ks;
    float shininess;
public:
    /**
    Material(vec3 f, vec3 b = vec3(0, 0, 0)) {
        frontColor = f;
        backColor = b;
    }**/
    
    Material(vec3 f, vec3 b = vec3(0, 0, 0), vec3 ks = vec3(0, 0, 0), float shininess = 1.0):
    frontColor(f), backColor(b), ks(ks), shininess(shininess) {}
    
    virtual vec3 getColor(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir)
    {
        float d = normal.dot(viewDir);
        if (d < 0) {
            return backColor * fabs(d);
        }
        else {
            return frontColor * d;
        }
    }
    
    virtual vec3 shade(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir,
                          vec3 lightDir,
                          vec3 powerDensity)
    {
        float d = normal.dot(viewDir);
        vec3 halfWay = (viewDir + lightDir).normalize();
        if (d < 0) {
            return powerDensity * backColor * fmax(0, normal.dot(lightDir)) + powerDensity * ks
            * pow(fmax(0, normal.dot(halfWay)), shininess);
        }
        else {
            return powerDensity * frontColor * fmax(0, normal.dot(lightDir)) + powerDensity * ks
            * pow(fmax(0, normal.dot(halfWay)), shininess);
        }
    }
};

float snoise(vec3 r) {
    unsigned int x = 0x0625DF73;
    unsigned int y = 0xD1B84B45;
    unsigned int z = 0x152AD8D0;
    float f = 0;
    for(int i=0; i<32; i++) {
        vec3 s(    x/(float)0xffffffff,
               y/(float)0xffffffff,
               z/(float)0xffffffff);
        f += sin(s.dot(r));
        x = x << 1 | x >> 31;
        y = y << 1 | y >> 31;
        z = z << 1 | z >> 31;
    }
    return f / 64.0 + 0.5;
}

vec3 noiseGrad(vec3 r) {
    vec3 s = vec3(7502, 22777, 4767);
    vec3 f = vec3(0.0, 0.0, 0.0);
    for(int i=0; i<16; i++) {
        f += (s - vec3(32768, 32768, 32768)) *
        cos( ((s - vec3(32768, 32768, 32768)).dot(r*40.0))
                 / 65536.0) * 40.0;
        s = vec3(fmod(s.x, 32768.0), fmod(s.y, 32768.0), fmod(s.z, 32768.0)) * 2.0 +
        vec3(floorf(s.x * (1/32768.0)), floorf(s.y * (1/32768.0)), floorf(s.z * (1/32768.0)));
    }
    return f * (1/65536.0);
}

class Marble : public Material
{
    vec3 color1, color2;
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    Marble(vec3 color1, vec3 color2, float scale):
    Material(vec3(1, 1, 1)), color1(color1), color2(color2), scale(scale)
    {
        turbulence = 50;
        period = 32;
        sharpness = 1;
    }
    virtual vec3 getColor(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir)
    {
        //return normal;
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
        w = pow(sin(w)*0.5+0.5, 4);
        return (color1 * w + color2 * (1-w)) * normal.dot(viewDir);
    }
    
    virtual vec3 shade(
                       vec3 position,
                       vec3 normal,
                       vec3 viewDir,
                       vec3 lightDir,
                       vec3 powerDensity)
    {
        float d = normal.dot(viewDir);
        vec3 halfWay = (viewDir + lightDir).normalize();
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
        w = pow(sin(w)*0.5+0.5, 4);
        if (d < 0) {
            return powerDensity * vec3(0, 0, 0) * fmax(0, normal.dot(lightDir));
        }
        else {
            return powerDensity * (color1 * w + color2 * (1-w)) * fmax(0, normal.dot(lightDir));
        }
    }
};

class Metal: public Material
{
    float reflectance;
public:
    Metal(): Material(vec3(0.08, 0.37, 0.37)) {
        reflectance = 0.4;
    }

    vec3 reflectionDir(vec3 normal, vec3 viewDir) {
        return (normal * normal.dot(viewDir) * 2 - viewDir).normalize();
    }
    
    float getReflectance() {
        return reflectance;
    }

    virtual vec3 shade(
                       vec3 position,
                       vec3 normal,
                       vec3 viewDir,
                       vec3 lightDir,
                       vec3 powerDensity)
    {
        float d = normal.dot(viewDir);
        vec3 halfWay = (viewDir + lightDir).normalize();
        if (d < 0) {
            return powerDensity * backColor * fabs(d) + powerDensity * ks
            * pow(fmax(0, normal.dot(halfWay)), shininess);
        }
        else {
            return powerDensity * frontColor * d + powerDensity * ks
            * pow(fmax(0, normal.dot(halfWay)), shininess);
        }
    }
};

class Glass: public Material
{
    float mu;
    float refractance;
public:
    Glass(): Material(vec3(1, 1, 1)) {
        mu = 0.4;
        refractance = 0.4;
    }
    
    vec3 refractionDir(vec3 normal, vec3 viewDir) {
        float cosalpha = normal.dot(viewDir);
        float sinalpha = sqrt(1-pow(cosalpha, 2));
        float sinbeta = sinalpha/mu;
        float cosbeta = sqrt(1-pow(sinbeta, 2));
        vec3 x = -normal * cosbeta;
        vec3 y = (normal * (normal.dot(viewDir)) - viewDir) * sinbeta;
        return (x + y).normalize();
    }
    
    float getRefractance() {
        return refractance;
    }
};

class Wood : public Material
{
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    Wood():
    Material(vec3(0, 1, 1))
    {
        scale = 26;
        turbulence = 400;
        period = 8;
        sharpness = 10;
    }
    virtual vec3 getColor(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir)
    {
        //return normal;
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
        w -= int(w);
        return (vec3(0.52, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1-w)) * normal.dot(viewDir);
    }
    
    virtual vec3 shade(
                       vec3 position,
                       vec3 normal,
                       vec3 viewDir,
                       vec3 lightDir,
                       vec3 powerDensity)
    {
        float d = normal.dot(viewDir);
        vec3 halfWay = (viewDir + lightDir).normalize();
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
        w -= int(w);
        if (d < 0) {
            return powerDensity * vec3(0, 0, 0) * fabs(d);
        }
        else {
            return powerDensity * (vec3(0.52, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1-w)) * d;
        }
    }
};

class Waves: public Material
{
    float scale;
public:
    Waves(): Material(vec3(1, 1, 1))
    {
        scale = 1;
    }
    virtual vec3 getColor(
                          vec3 position,
                          vec3 normal,
                          vec3 viewDir)
    {
        
        float d = normal.dot(viewDir);
        vec3 noiseG = noiseGrad(position * scale);
        //return normal
        return (noiseG).normalize();
    }
};

class LightSource
{
protected:
    vec3 powerDensity;
public:
    LightSource(vec3 powerDensity): powerDensity(powerDensity){}
    virtual vec3 getPowerDensityAt ( vec3 x )=0;
    virtual vec3 getLightDirAt     ( vec3 x )=0;
    virtual float  getDistanceFrom ( vec3 x )=0;
};

class DirectionalLightSource : public LightSource
{
    vec3 lightDirection;
public:
    DirectionalLightSource(vec3 lightDirection, vec3 powerDensity = vec3(1, 1, 1)):
    LightSource(powerDensity), lightDirection(lightDirection) {}
    
    vec3 getPowerDensityAt ( vec3 x ) {
        return powerDensity;
    }
    
    vec3 getLightDirAt ( vec3 x ) {
        return lightDirection.normalize();
    }
    float getDistanceFrom ( vec3 x ) {
        return 1000000.0;
    }
};

class PointLightSource : public LightSource
{
    vec3 lightPosition;
public:
    PointLightSource(vec3 lightPosition, vec3 powerDensity = vec3(1, 1, 1)):
    LightSource(powerDensity), lightPosition(lightPosition) {}
    
    vec3 getPowerDensityAt ( vec3 x ) {
        return powerDensity * (1/(x - lightPosition).norm2());
    }
    
    vec3 getLightDirAt ( vec3 x ) {
        return (lightPosition - x).normalize();
    }
    float getDistanceFrom ( vec3 x ) {
        return (x - lightPosition).norm();
    }
};

class Camera
{
    vec3 eye;        // World space camera position.
    vec3 lookAt;    // Center of window in world space.
    vec3 right;        // Vector from window center to window right-mid (in world space).
    vec3 up;        // Vector from window center to window top-mid (in world space).
    
public:
    Camera()
    {
        eye = vec3(0, 0.5, 2);
        //lookAt = vec3(-0.5, 0.8, 1);
        lookAt = vec3(-0.2, 0.35, 1);
        right = vec3(1, 0, 0);
        up = vec3(0, 1, 0);
    }
    vec3 getEye()
    {
        return eye;
    }
    
    // Compute ray through pixel at normalized device coordinates.
    
    vec3 rayDirFromNdc(float x, float y) {
        return (lookAt - eye
                + right * x
                + up    * y
                ).normalize();
    }
};

// Ray structure.

class Ray
{
public:
    vec3 origin;
    vec3 dir;
    Ray(vec3 o, vec3 d)
    {
        origin = o;
        dir = d;
    }
};

// Hit record structure. Contains all data that describes a ray-object intersection point.

class Hit
{
public:
    Hit()
    {
        t = -1;
    }
    float t;                // Ray paramter at intersection. Negative means no valid intersection.
    vec3 position;            // Intersection coordinates.
    vec3 normal;            // Surface normal at intersection.
    Material* material;        // Material of intersected surface.
};

// Abstract base class.

class Intersectable
{
protected:
    Material* material;
public:
    Intersectable(Material* material):material(material) {}
    virtual Hit intersect(const Ray& ray)=0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.

class QuadraticRoots
{
public:
    float t1;
    float t2;
    
    // Solves the quadratic a*t*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and sets members t1 and t2 to store the roots.
    
    QuadraticRoots(float a, float b, float c)
    {
        float discr = b * b - 4.0 * a * c;
        if ( discr < 0 ) // no roots
        {
            t1 = -1;
            t2 = -1;
            return;
        }
        float sqrt_discr = sqrt( discr );
        t1 = (-b + sqrt_discr)/2.0/a;
        t2 = (-b - sqrt_discr)/2.0/a;
    }
    
    // Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
    
    float getLesserPositive()
    {
        return (0 < t1 && (t2 < 0 || t1 < t2)) ? t1 : t2;
    }
};

// Object realization.

class Sphere : public Intersectable
{
    vec3 center;
    float radius;
public:
    Sphere(const vec3& center, float radius, Material* material):
    Intersectable(material),
    center(center),
    radius(radius)
    {
    }
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        vec3 diff = ray.origin - center;
        float a = ray.dir.dot(ray.dir);
        float b = diff.dot(ray.dir) * 2.0;
        float c = diff.dot(diff) - radius * radius;
        return QuadraticRoots(a, b, c);
        
    }
    vec3 getNormalAt(vec3 r)
    {
        return (r - center).normalize();
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        float t = solveQuadratic(ray).getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
};

class Plane : public Intersectable
{
    vec3 center, normal;
    
public:
    Plane(const vec3& center, const vec3& normal, Material* material):
    Intersectable(material),
    center(center),
    normal(normal)
    {}
    
    vec3 getNormalAt(vec3 r)
    {
        return normal;
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        float d = normal.dot(ray.dir);
        
        Hit hit;
        if (d == 0.0) return hit;
        
        float t = normal.dot(center - ray.origin) / d;
        
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
};

class Quadric : public Intersectable
{
protected:
    mat4x4 A;
    
public:
    Quadric(Material* material): Intersectable(material)
    {
        //default constructor is sphere
        A._33 = -1;
    }
    
    bool isDefault() {
        return A._33 == -1;
    }
    
    Quadric* cylinder()
    {
        A._11 = 0;
        A._33 = -1;
        return this;
    }
    
    Quadric* cone()
    {
        A._00 = 1;
        A._11 = -1;
        A._22 = 1;
        A._33 = 0;
        return this;
    }
    
    Quadric* paraboloid()
    {
        A._11 = 0;
        A._33 = 0;
        A._13 = -1;
        return this;
    }
    
    Quadric* parallelPlanes(float height) {
        A._00 = 0;
        A._11 = 1;
        A._22 = 0;
        A._33 = -((height*height)/4);
        return this;
    }
    
    Quadric* parallelPlanes2(float height) {
        A._00 = 0;
        A._11 = 0;
        A._22 = 1;
        A._33 = -((height*height)/4);
        return this;
    }
    
    Quadric* parallelPlanes3(float height) {
        A._00 = 1;
        A._11 = 0;
        A._22 = 0;
        A._33 = -((height*height)/4);
        return this;
    }
    
    Quadric* transform(mat4x4 T) {
        A = T.invert() * A * (T.invert()).transpose();
        return this;
    }
    
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        // direction -> add 0, position -> add 1
        vec4 e = ray.origin; //default is 1
        vec4 d = ray.dir; d.w = 0.0;
        
        float a = d.dot(A * d);
        float b = d.dot(A * e) + e.dot(A * d);
        float c = e.dot(A * e);
        return QuadraticRoots(a, b, c);
    }
    
    bool contains(vec3 r) {
        vec4 x = r;
        if (x.dot(A * x) < 0.0) return true;
        else return false;
    }
    
    vec3 getNormalAt(vec3 r)
    {
        vec4 x = r;
        vec4 n = A * x + x * A;
        return vec3(n.x, n.y, n.z).normalize();
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        float t = solveQuadratic(ray).getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
};

class ClippedQuadric : public Intersectable
{
    Quadric shape, clipper, clipper2;
    
public:
    ClippedQuadric(Material* material): Intersectable(material), shape(material), clipper(material),
    clipper2(material){}
    
    ClippedQuadric* cylinder(float height)
    {
        shape.cylinder();
        clipper.parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* cone(float height)
    {
        shape.cone();
        //translate by height/2
        clipper.parallelPlanes(height)->transform(mat4x4::translation(vec3(0, -height/2, 0)));
        return this;
    }
    
    ClippedQuadric* paraboloid(float height)
    {
        shape.paraboloid();
        clipper.parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* slab1(float height)
    {
        shape.parallelPlanes(height);
        clipper.parallelPlanes3(height);
        clipper2.parallelPlanes2(height);
        return this;
    }
    
    ClippedQuadric* slab2(float height)
    {
        shape.parallelPlanes3(height);
        clipper.parallelPlanes2(height);
        clipper2.parallelPlanes(height);
        return this;
    }
    
    ClippedQuadric* slab3(float height)
    {
        shape.parallelPlanes2(height);
        clipper.parallelPlanes(height);
        clipper2.parallelPlanes3(height);
        return this;
    }
    
    ClippedQuadric* transform(mat4x4 t) {
        shape.transform(t);
        clipper.transform(t);
        return this;
    }
    
    vec3 getNormalAt(vec3 r)
    {
        return shape.getNormalAt(r);
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal.
        
        QuadraticRoots roots = shape.solveQuadratic(ray);
        
        vec3 p1 = ray.origin + ray.dir * roots.t1;
        vec3 p2 = ray.origin + ray.dir * roots.t2;
        
        if (!clipper.contains(p1)) roots.t1 = -1;
        if (!clipper.contains(p2)) roots.t2 = -1;
        if (!clipper2.isDefault() && !clipper2.contains(p1)) roots.t1 = -1;
        if (!clipper2.isDefault() && !clipper2.contains(p2)) roots.t2 = -1;
        float t = roots.getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
};

class Scene
{
    Camera camera;
    //Sphere sphere;        // THIS NEEDS TO GO WHEN YOU USE A VECTOR OF INTERSECTABLE OBJECTS
    std::vector<Intersectable*> objects;
    //Material material;    // THIS NEEDS TO GO WHEN YOU USE A VECTOR OF MATERIALS
    std::vector<Material*> materials;
    std::vector<LightSource*> lights;
public:
    Scene()
    //: sphere( vec3(0, 0, 0), 1, &material)    // THIS NEEDS TO GO WHEN THE SPHERE OBJECT IS GONE
    {
        lights.push_back(new DirectionalLightSource(vec3(1, 2, 0)));
        //lights.push_back(new DirectionalLightSource(vec3(-0.2, 0.35, 1)));
        lights.push_back(new PointLightSource(vec3(0, 2, -1)));
        //lights.push_back(new PointLightSource(vec3(1, 1, 2)));
        
        materials.push_back(new Material(vec3(1, 0.5, 0.5)));//pink
        materials.push_back(new Material(vec3(0.66, 0.59, 0.39)));//yellow
        materials.push_back(new Material(vec3(0, 0.30, 0.42), vec3(0, 0, 0), vec3(1, 1, 1), 100)); //blue
        materials.push_back(new Material(vec3(1, 1, 1)));
        materials.push_back(new Marble(vec3(1, 1, 1), vec3(1, 0.5, 0), 2));
        materials.push_back(new Wood());
        materials.push_back(new Waves());
        materials.push_back(new Material(vec3(0.08, 0.37, 0.37)));//green
        materials.push_back(new Metal());
        
        float forward = 0.2;
        float left = 1.1;
        float up = 0.2;
        
        //sand castle
        objects.push_back((new ClippedQuadric(materials[1]))->cone(2)->transform(
                mat4x4::scaling(vec3(0.05, 0.05, 0.05))
              * mat4x4::translation(vec3(-0.1-left, 0.15+up, 0.3-forward))));
        objects.push_back((new ClippedQuadric(materials[1]))->cylinder(2)->transform(
                mat4x4::scaling(vec3(0.1,0.15,0.1))
              * mat4x4::translation(vec3(-0.1-left, -0.1+up, 0.3-forward))));
        objects.push_back((new ClippedQuadric(materials[1]))->cone(2)->transform(
                mat4x4::scaling(vec3(0.05, 0.05, 0.05))
              * mat4x4::translation(vec3(-0.5-left, 0.15+up, 0.3-forward))));
              //* mat4x4::rotation(vec3(1, 0, 0), 90)));
        objects.push_back((new ClippedQuadric(materials[1]))->cylinder(2)->transform(
                mat4x4::scaling(vec3(0.1,0.15,0.1))
              * mat4x4::translation(vec3(-0.5-left, -0.1+up, 0.3-forward))));
        objects.push_back((new ClippedQuadric(materials[1]))->cylinder(2)->transform(
                mat4x4::scaling(vec3(0.10,0.2,0.10))
              * mat4x4::rotation(vec3(0,0,1), 90*(M_PI/180))
              * mat4x4::translation(vec3(-0.25-left, -0.18+up, 0.3-forward))));
        
        //water
        objects.push_back((new Quadric(materials[8]))->parallelPlanes(2)->transform(
                mat4x4::translation(vec3(0, -1.4, 0))));
        //objects.push_back(new Plane(vec3(0, -0.9, 0), vec3(0, 1, 0), materials[6]));
        
        //island
        objects.push_back((new ClippedQuadric(materials[1]))->paraboloid(2)->transform(
                mat4x4::scaling(vec3(3.5, 0.5, 1))
              * mat4x4::rotation(vec3(1,0,0), 180)
              * mat4x4::translation(vec3(-2, -0.4, 0.9))));
        
        //ball
        objects.push_back((new Quadric(materials[4]))->transform(
                mat4x4::scaling(vec3(0.2, 0.2, 0.2))
              * mat4x4::translation(vec3(-0.5, -0.3, 1.2))));
        
        //box
        objects.push_back((new ClippedQuadric(materials[5]))->slab1(0.4)->transform(
                mat4x4::scaling(vec3(1,1,1))
              * mat4x4::rotation(vec3(0,0,0), 30*(M_PI/180))
              * mat4x4::translation(vec3(0, 0, 0))));
        
        objects.push_back((new ClippedQuadric(materials[5]))->slab2(0.4)->transform(
                mat4x4::scaling(vec3(1,1,1))
              * mat4x4::rotation(vec3(0,0,0), 30*(M_PI/180))
              * mat4x4::translation(vec3(0, 0, 0))));
        
        objects.push_back((new ClippedQuadric(materials[5]))->slab3(0.4)->transform(
                mat4x4::scaling(vec3(1,1,1))
              * mat4x4::rotation(vec3(0,0,0), 30*(M_PI/180))
              * mat4x4::translation(vec3(0, 0, 0))));
        
        
        //parasol
        
        objects.push_back((new ClippedQuadric(materials[3]))->cylinder(8)->transform(
            mat4x4::scaling(vec3(0.03, 0.6, 0.03))
          * mat4x4::rotation(vec3(0,0,1), 10*(M_PI/180))
          * mat4x4::translation(vec3(-1.4, 0, 0.4))));
        objects.push_back((new ClippedQuadric(materials[0]))->paraboloid(1)->transform(
            mat4x4::scaling(vec3(1.3,1.3,1.3))
          * mat4x4::rotation(vec3(1,0,0), 230*(M_PI/180))
          * mat4x4::translation(vec3(-1.7,2.4,0.5))));
        
        
        objects.push_back((new Quadric(materials[2]))->transform(
                mat4x4::scaling(vec3(0.2, 0.2, 0.2))
              * mat4x4::translation(vec3(0.4, 0.3, 1))));
    
    }
    ~Scene()
    {
        // UNCOMMENT THESE WHEN APPROPRIATE
        for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
        delete *iMaterial;
        for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
        delete *iObject;
        for (std::vector<LightSource*>::iterator iLight = lights.begin(); iLight != lights.end(); ++iLight)
        delete *iLight;
    }
    
public:
    Camera& getCamera()
    {
        return camera;
    }
    
    Hit firstIntersect(const Ray& ray)
    {
        Hit firstHit;
        float tmin = 1000000.0;
        for (int i = 0; i < objects.size(); i++) {
            Hit hit = objects[i]->intersect(ray);
            if (hit.t > 0 && hit.t < tmin) {
                tmin = hit.t;
                firstHit = hit;
            }
        }
        return firstHit;
    }
    
    vec3 trace(const Ray& ray, int depth = 3)
    {
        const float epsilon = 0.001;
        
        //Hit hit = sphere.intersect(ray);
        Hit hit = firstIntersect(ray);
        if(hit.t < 0)
        return vec3(1, 0.97, 0.92); //sky color
        
        vec3 color;
        
        for (int i = 0; i < lights.size(); i++) {
            vec3 lightDir = lights[i]->getLightDirAt(hit.position);
            Ray shadowRay(hit.position + hit.normal * epsilon, lightDir);
            Hit shadowHit = firstIntersect(shadowRay);
            if ((shadowHit.t < 0.0) || (shadowHit.t > lights[i]->getDistanceFrom(hit.position))) {
                //color += hit.material->getColor(hit.position, hit.normal, lightDir);
                color += hit.material->shade(hit.position, hit.normal, -ray.dir, lightDir, lights[i]->getPowerDensityAt(hit.position));
            }
        }
        
        Metal *metal = dynamic_cast< Metal*>(hit.material);
        Glass *glass = dynamic_cast< Glass*>(hit.material);
        if(metal) {
            vec3 reflectionDir = metal->reflectionDir(hit.normal, -ray.dir);
            Ray reflectedRay(hit.position + hit.normal * epsilon, reflectionDir);
            if (depth > 0) {
               color += trace(reflectedRay, depth - 1) * metal->getReflectance();
            }
        }
        if(glass) {
            vec3 refractionDir = glass->refractionDir(hit.normal, -ray.dir);
            Ray refractedRay(hit.position + hit.normal * epsilon, refractionDir);
            if (depth > 0) {
                color += trace(refractedRay, depth - 1) * glass->getRefractance();
            }
        }
        
        
        return color; //hit.material->getColor(hit.position, hit.normal, -ray.dir);
    }
};

Scene scene;




class FrameBuffer {
    unsigned int textureId;
    vec3 image[windowWidth * windowHeight];
    
public:
    FrameBuffer() {
        for(int i = 0; i < windowWidth * windowHeight; i++) image[i] = vec3(0.0, 0.0, 0.0);
        
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    
    void Bind(Shader* s)
    {
        s->UploadSamplerID();
        glBindTexture(GL_TEXTURE_2D, textureId);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image);
    }
    
    bool ComputeImage()
    {
        static unsigned int iPart = 0;
        
        if(iPart >= 64)
        return false;
        for(int j = iPart; j < windowHeight; j+=64)
        {
            for(int i = 0; i < windowWidth; i++)
            {
                float ndcX = (2.0 * i - windowWidth) / windowWidth;
                float ndcY = (2.0 * j - windowHeight) / windowHeight;
                Camera& camera = scene.getCamera();
                Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcX, ndcY));
                
                image[j*windowWidth + i] = scene.trace(ray);
            }
        }
        iPart++;
        return true;
    }
};

class Screen {
    FrameBuffer frameBuffer;
    unsigned int vao;
    
public:
    Screen()
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        unsigned int vbo[2];
        glGenBuffers(2, &vbo[0]);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        static float vertexCoords[] = { -1, -1,        1, -1,        -1, 1,        1, 1 };
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        static float vertexTextureCoords[] = { 0, 0,    1, 0,        0, 1,        1, 1 };
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTextureCoords), vertexTextureCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }
    
    void Draw(Shader* s)
    {
        if(frameBuffer.ComputeImage())
        glutPostRedisplay();
        
        s->Run();
        frameBuffer.Bind(s);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisable(GL_BLEND);
    }
};

Screen *screen = 0;


void onDisplay( ) {
    //glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClearColor(0.3, 0.48, 0.52, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    screen->Draw(shader);
    
    glutSwapBuffers();
}

void onInitialization()
{
    glViewport(0, 0, windowWidth, windowHeight);
    
    shader = new Shader();
    
    screen = new Screen();
}

void onExit()
{
    delete screen; screen = 0;
    delete shader; shader = 0;
    printf("exit");
}

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(10, 10);
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow("Ray Casting");
    
#if !defined(__APPLE__)
    glewExperimental = true;
    glewInit();
#endif
    
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    glViewport(0, 0, windowWidth, windowHeight);
    
    onInitialization();
    
    glutDisplayFunc(onDisplay);
    
    glutMainLoop();
    
    onExit();
    
    return 1;
}




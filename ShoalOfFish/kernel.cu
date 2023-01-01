
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <cmath>
#define width 1280   //screen width
#define height 720   //screen height
#define fishNumber 10
#define maxThreds 1024
#define maxBlocks 100
#define M_PI 3.14159265358979323846

struct Shoal {
    int* position_x;
    int* position_y;
    float* direction;
    int h = 30;
    int w = 6;
    int distance = 20;
};

struct Point {
    int x;
    int y;
};

int x = 0;
float t = 0.0f;
float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer
Shoal shoal;
float background;
float fishColor;

__device__ int sign(Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ bool ContainsPixel(int x, int y, int fishX, int fishY, float direction, int h, int w)
{
    Point pixel;
    pixel.x = x; pixel.y = y;
    Point p1, p2, p3;

    int hX = cos(direction)* h / 2 / sqrt((float)2);
    int hY = sin(direction)* h / 2 / sqrt((float)2);

    p1.x = fishX + hX;
    p1.y = fishY + hY;
    
    int fishWingX = sin(direction) * w / sqrt((float)2);
    int fishWingY = cos(direction) * w / sqrt((float)2);

    p2.x = fishX - hX + fishWingX;
    p2.y = fishY - hY + fishWingY;

    p3.x = fishX - hX - fishWingX;
    p3.y = fishY - hY - fishWingY;

    bool has_neg, has_pos;

    int d1 = sign(pixel, p1, p2);
    int d2 = sign(pixel, p2, p3);
    int d3 = sign(pixel, p3, p1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

__device__ __host__ float ConvertToFloat(float3 c)
{
    float colour;
    unsigned char bytes[] = { (unsigned char)(c.x * 255 + 0.5), (unsigned char)(c.y * 255 + 0.5), (unsigned char)(c.z * 255 + 0.5), 1 };
    memcpy(&colour, &bytes, sizeof(colour));
    return colour;
}
__global__ void CalculateShoal(Shoal shoal) 
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > fishNumber)
        return;
    float direction = shoal.direction[x];
    direction += M_PI / 40;
    if (direction > 2 * M_PI)
        direction - 2 * M_PI;
    else if(direction < 0)
        direction + 2 * M_PI;

    shoal.direction[x] = direction;
    
}
__global__ void Drawe(float3* output, int k, Shoal shoal, float background, float fishColor)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = (height - y - 1) * width + x;

    for (int f = 0; f < fishNumber; f++)
    {
        if (ContainsPixel(x, y, shoal.position_x[f], shoal.position_y[f], shoal.direction[f], shoal.h, shoal.w))
        {
            output[i] = make_float3(x, y, fishColor);
            return;
        }
    }

    output[i] = make_float3(x, y, background);
}


__global__ void InitStartPosition(Shoal shoal, int fisheGrideWidth, int fishGrideHeight)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    int distance = shoal.distance;
    int startPoint_x = (width - distance*fisheGrideWidth) / 2;
    int startPoimt_y = (height - distance*fishGrideHeight) / 2;
    int fishGridY = x / fisheGrideWidth;
    int fishGridX = x - fishGridY * fisheGrideWidth;
    shoal.direction[x] = M_PI / 2;
    shoal.position_x[x] = startPoint_x + fishGridX * distance;
    shoal.position_y[x] = startPoimt_y + fishGridY * distance;
}

// Global variables to track the position of the dot


void time(int x)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(10, time, 0);
        t += 0.0166f;
    }
}

void CalculateNeededThreads(int* threads, int* blocks)
{
    if (fishNumber < maxThreds)
    {
        *threads = fishNumber;
        *blocks = 1;
    }
    else
    {
        *threads = maxThreds;
        *blocks = ceil(fishNumber / maxThreds);
        if (*blocks > maxBlocks)
            exit(-1);
    }
}

void LunchCuda()
{
    int blocks, threads;
    CalculateNeededThreads(&threads, &blocks);
    CalculateShoal << <blocks, threads >> > (shoal);
    cudaThreadSynchronize();

    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    Drawe << < grid, block >> > (device, x, shoal, background, fishColor);   //execute kernel
}
// Display callback function
void display() {
    // Clear the window
    cudaThreadSynchronize();
    cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
    glClear(GL_COLOR_BUFFER_BIT);

    LunchCuda();
    cudaThreadSynchronize();

    cudaGLUnmapBufferObject(buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glVertexPointer(2, GL_FLOAT, 12, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();
    x++;
    if (x > width)
        x = 0;
}

void InitCuda()
{
    cudaMalloc(&device, width * height * sizeof(float3));   //allocate memory on the GPU VRAM
    cudaMalloc(&shoal.position_x, fishNumber * sizeof(int));
    cudaMalloc(&shoal.position_y, fishNumber * sizeof(float));
    cudaMalloc(&shoal.direction, fishNumber * sizeof(float));

    int fisheGrideWidth = ceil(sqrt((fishNumber * width) / height));
    int fishGrideHeight = ceil(fisheGrideWidth * height / width);

    int blocks, threads;
    CalculateNeededThreads(&threads, &blocks);
    InitStartPosition << <blocks, threads >> > (shoal, fisheGrideWidth, fishGrideHeight);
}

void Init()
{
    background = ConvertToFloat(make_float3(0.0, 0.0, 0.0));
    fishColor = ConvertToFloat(make_float3(0.2, 0.8, 0.2));
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width, 0.0, height);
    glutDisplayFunc(display);
    //glutReshapeFunc(Reshape);
    time(0);
    glewInit();
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    unsigned int size = width * height * sizeof(float3);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    InitCuda();
    cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);   //display mode
    glutInitWindowSize(width, height);
    glutCreateWindow("Moving Dot"); // Create the window
    Init();
    // Enter the main loop
    glutMainLoop();
    cudaFree(device);
    return 0;
}
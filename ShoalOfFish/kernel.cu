// ilość ryb w kompilatorze ustalana 
// pomysł z tablicą na karzdy sektor jest słaby
// trzeba jakość segregować rybki w zależności od grupy
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <cmath>
#define width 1280   //screen width
#define height 700   //screen height
#define fishNumber 1000
#define maxThreds 1024
#define maxBlocks 100
#define M_PI 3.14159265358979323846
#define MaxSpeed 10

constexpr float CohesionScale = 0.01f;
constexpr float AlignmentScale = 0.1f;
constexpr float SeparationScale = 0.1f;


struct Shoal {
    float* position_x;
    float* position_y;
    float* velocity_x;
    float* velocity_y;
    int h = 20;
    int w = 5;
    int minDistance = 20;
    int viewRange = 100;
};

struct Point {
    double x;
    double y;
};

int x = 0;
float t = 0.0f;
float* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer
Shoal shoal;
float background;
float fishColor;

__device__ int sign(Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ Point MakePoint(float x, float y)
{
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

__device__ double Distance(Point p1, Point p2)
{
    return pow(p1.x - p2.x, (double)2) + pow(p1.y - p2.y, (double)2);
}

__device__ float Direction(Point vel)
{
    float direction = asin(vel.y / sqrt(Distance(MakePoint(0, 0), vel)));
    
    if (vel.x < 0)
    {
        if (vel.y >= 0)
        {
            float alfa = M_PI / 2 - direction;
            direction += 2 * alfa;
        }
        else
        {
            float alfa = M_PI / 2 + direction;
            direction -= 2 * alfa;
        }
    }
    return direction;
}

__device__ void FishToCordinates(Point fishPosition, float direction, int h, int w, Point* p1, Point* p2, Point* p3)
{
    float hX = cos(direction) * h / 2;
    float hY = sin(direction) * h / 2;

    p1->x = fishPosition.x + hX;
    p1->y = fishPosition.y + hY;

    float fishWingX = -sin(direction) * w;
    float fishWingY = cos(direction) * w;

    p2->x = fishPosition.x - hX + fishWingX;
    p2->y = fishPosition.y - hY + fishWingY;

    p3->x = fishPosition.x - hX - fishWingX;
    p3->y = fishPosition.y - hY - fishWingY;
}
__global__ void CalculateShoal(Shoal shoal, float* output)
{
    unsigned int fishId = blockIdx.x * blockDim.x + threadIdx.x;
    if (fishId > fishNumber)
        return;

    float viewRange = shoal.viewRange * shoal.viewRange;
    float minDistance = shoal.minDistance * shoal.minDistance;
    
    
        Point bois = MakePoint(shoal.position_x[fishId], shoal.position_y[fishId]);
        Point centerOfMass = MakePoint(0,0);
        Point separation = MakePoint(0, 0);
        Point avrVelocity = MakePoint(0, 0);
        int neighboursCountCenterOfMass = 0;
        int neighboursCountVelocity = 0;
        for (int neighbourFishId = 0; neighbourFishId < fishNumber; neighbourFishId++)
        {
            if (neighbourFishId == fishId)
                continue;

            Point boisFriend = MakePoint(shoal.position_x[neighbourFishId], shoal.position_y[neighbourFishId]);
            double distance = Distance(bois, boisFriend);
            if (viewRange > distance)
            {
                neighboursCountCenterOfMass++;
                centerOfMass.x += boisFriend.x;
                centerOfMass.y += boisFriend.y;

            }

            if (viewRange > distance)
            {
                neighboursCountVelocity++;
                avrVelocity.x += shoal.velocity_x[neighbourFishId];
                avrVelocity.y += shoal.velocity_x[neighbourFishId];
            }
            if (minDistance > distance)
            {
                separation.x -= boisFriend.x - bois.x;
                separation.y -= boisFriend.y - bois.y;
            }
        }
        Point newPosition = MakePoint(shoal.velocity_x[fishId], shoal.velocity_y[fishId]);
        if (neighboursCountCenterOfMass > 0)
        {
            centerOfMass.x /= neighboursCountCenterOfMass;
            centerOfMass.y /= neighboursCountCenterOfMass;

             newPosition.x += (centerOfMass.x - bois.x) * CohesionScale;
             newPosition.y += (centerOfMass.y - bois.y) * CohesionScale;
        }

        separation.x *= SeparationScale;
        separation.y *= SeparationScale;

        newPosition.x += separation.x;
        newPosition.y += separation.y;

        if (neighboursCountVelocity != 0) {
            avrVelocity.x /= neighboursCountVelocity;
            avrVelocity.y /= neighboursCountVelocity;
            newPosition.x += avrVelocity.x * AlignmentScale;
            newPosition.y += avrVelocity.y * AlignmentScale;
        }

        if (Distance(MakePoint(0, 0), newPosition) > MaxSpeed * MaxSpeed)
        {
            double calculateSpeed = sqrt( Distance(MakePoint(0, 0), newPosition));
            newPosition.x *= MaxSpeed / calculateSpeed;
            newPosition.y *= MaxSpeed / calculateSpeed;
        }
    

    shoal.position_x[fishId] += newPosition.x;
    shoal.position_y[fishId] += newPosition.y;
    shoal.velocity_x[fishId] = newPosition.x;
    shoal.velocity_y[fishId] = newPosition.y;

    if (shoal.position_x[fishId] < 0)
        shoal.position_x[fishId] = width;
    else if (shoal.position_x[fishId] > width)
        shoal.position_x[fishId] = 0;

    if (shoal.position_y[fishId] < 0)
        shoal.position_y[fishId] = height;
    else if (shoal.position_y[fishId] > height)
        shoal.position_y[fishId] = 0;

    Point p1, p2, p3;
    FishToCordinates(bois, Direction(newPosition), shoal.h, shoal.w, &p1, &p2, &p3);
    int start = fishId * 6;
    output[start] = p1.x;
    output[start + 1] = p1.y;
    output[start + 2] = p2.x;
    output[start + 3] = p2.y;
    output[start + 4] = p3.x;
    output[start + 5] = p3.y;
}



__global__ void InitStartPosition(Shoal shoal, int fisheGrideWidth, int fishGrideHeight)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    int distance = shoal.minDistance;
    int startPoint_x = (width - distance*fisheGrideWidth) / 2;
    int startPoimt_y = (height - distance*fishGrideHeight) / 2;
    int fishGridY = x / fisheGrideWidth;
    int fishGridX = x - fishGridY * fisheGrideWidth;
    shoal.velocity_x[x] = MaxSpeed * cos(x * 2* M_PI / fishNumber);
    shoal.velocity_y[x] = MaxSpeed * sin(x * 2 * M_PI / fishNumber);
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
    CalculateShoal << <blocks, threads >> > (shoal, device);
    cudaThreadSynchronize();

}
// Display callback function
void display() {
    // Clear the window
    cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
    glClear(GL_COLOR_BUFFER_BIT);

    LunchCuda();
    cudaThreadSynchronize();

    cudaGLUnmapBufferObject(buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    /*glVertexPointer(2, GL_FLOAT, 12, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);*/
    glDrawArrays(GL_TRIANGLES, 0, 3*fishNumber);
    glutSwapBuffers();
    
    x++;
    if (x > width)
        x = 0;
}

void InitCuda()
{
    cudaMalloc(&device, fishNumber * 6 * sizeof(float));   //allocate memory on the GPU VRAM
    cudaMalloc(&shoal.position_x, fishNumber * sizeof(float));
    cudaMalloc(&shoal.position_y, fishNumber * sizeof(float));
    cudaMalloc(&shoal.velocity_x, fishNumber * sizeof(float));
    cudaMalloc(&shoal.velocity_y, fishNumber * sizeof(float));
    int fisheGrideWidth = ceil(sqrt((fishNumber * width) / height));
    int fishGrideHeight = ceil(fisheGrideWidth * height / width);

    int blocks, threads;
    CalculateNeededThreads(&threads, &blocks);
    InitStartPosition << <blocks, threads >> > (shoal, fisheGrideWidth, fishGrideHeight);
}

void Init()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width, 0.0, height);
    glutDisplayFunc(display);
    //glutReshapeFunc(Reshape);
    time(0);
    glewInit();
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    unsigned int size = fishNumber * 6 * sizeof(float); // ilość wektorów
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
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
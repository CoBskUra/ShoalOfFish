﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <cmath>
#include <string>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <fstream>
#define width 1200   //screen width
#define height 750   //screen height
#define fishNumber 10000
#define maxThreds 512
#define maxBlocks 100
#define M_PI 3.14159265358979323846
#define MaxSpeed 5
#define Timer 10
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
constexpr float CohesionScale = 0.01f;
constexpr float AlignmentScale = 0.1f;
constexpr float SeparationScale = 0.1f;

struct Shoal {
    float* position_x;
    float* position_y;
    float* velocity_x;
    float* velocity_y;
    int h = 5;
    int w = 1;
    int minDistance = 20;
    int viewRange = 80;
};

struct Grid {
    int* cellsId;
    int* firstFishInCell;
    int* lastFishInCell;
    int* gridMapper;
    int gridNumber_Vertical;
    int gridNumber_Horyzontal;
    int gridWidth = 60;
    int gridHeight = 60;
};

struct Point {
    double x;
    double y;
};


float* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer
Shoal shoal;
Shoal scatterShoal;
Grid grid;
long long int lastTime;
int nbFrames = 0;

std::string readFileIntoString(const std::string& path);
void FreeShoalOfFish();
static unsigned int CompileShader(unsigned int type, const std::string& source);
static unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);
__device__ Point MakePoint(float x, float y);
__device__ double PowDistance(Point p1, Point p2); // zwraca kwadrat odległości między punktmi
__device__ float Direction(Point vel); // zwraca kąt reprezentowany przez wektor
__device__ void FishToCordinates(Point fishPosition, float direction,
    int h, int w, Point* p1, Point* p2, Point* p3);// przekształca pozycje rybki oraz 
                                                                       // kierunek na reprezentujący rybę trójkąt
__device__ int FishsCellId(Point fish, int gridWidth, 
    int gridHeight, int gridNumber_Horyzontal); // zwraca komórkę w jakiej znajduje się rybka
__global__ void CategorizeFishToCells(Shoal shoal, Grid grid); // przyporządkowuje rybką komórki
__global__ void CalculateShoal(Shoal shoal, Grid grid, float* output);// oblicza nowe położenie rybek
__global__ void InitStartPosition(Shoal shoal, int fisheGrideWidth, int fishGrideHeight); // inicjuje pozycje startową 
__global__ void ResetGridStartEnd(int* start, int* end, int size);// ustawia wartości tablic jako -1
__global__ void CalculateStartEnd(int* start, int* end, int* gridId);// oblicza id pierwszej i ostatniej rybki w danej komórce
__global__ void ResetMapper(int* mapper, int size); // restartuje mapper
__global__ void MappShoal(Shoal shoal, Shoal tmpShoal, Grid grid, int size);
__global__ void CopyShoal(Shoal shoal, Shoal tmpShoal, int size);
void CalculateNeededThreads(int* threads, int* blocks, int neededThreads);// oblicza ilość potrzebnych bloków i ich wielkość
void LunchCuda(); // uruchamia obliczenia na kernelu
void InitCuda(); // allokuje pamięć karty
void display();
void time(int x);
void Init();
void FreeShoalOfFish(); // zwalnia pamięć
void FrameCounter();


int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);   //display mode
    glutInitWindowSize(width, height);
    glutCreateWindow("ShoalOfFish"); // Create the window
    Init();
    // Enter the main loop
    glutMainLoop();
    FreeShoalOfFish();
    return 0;
}


std::string readFileIntoString(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
            << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}


static unsigned int CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;

}

static unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

__device__ Point MakePoint(float x, float y)
{
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

__device__ double PowDistance(Point p1, Point p2)
{
    return pow(p1.x - p2.x, (double)2) + pow(p1.y - p2.y, (double)2);
}

__device__ float Direction(Point vel)
{
    float direction = asin(vel.y / sqrt(PowDistance(MakePoint(0, 0), vel)));

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


__device__ int FishsCellId(Point fish, int gridWidth, int gridHeight, int gridNumber_Horyzontal)
{
    int gridX = fish.x / gridWidth;
    int gridY = fish.y / gridHeight;
    return gridY * gridNumber_Horyzontal + gridX;
}

__global__ void CategorizeFishToCells(Shoal shoal, Grid grid)
{
    unsigned int fishId = blockIdx.x * blockDim.x + threadIdx.x;
    if (fishId >= fishNumber)
        return;

    Point fish = MakePoint(shoal.position_x[fishId], shoal.position_y[fishId]);

    grid.cellsId[fishId] = FishsCellId(fish, grid.gridWidth, grid.gridHeight, grid.gridNumber_Horyzontal);
}




__global__ void CalculateShoal(Shoal shoal, Grid grid, float* output)
{
    unsigned int fishId = blockIdx.x * blockDim.x + threadIdx.x;
    if (fishId >= fishNumber)
        return;

    int powViewRange = shoal.viewRange * shoal.viewRange;
    int powMinDistance = shoal.minDistance * shoal.minDistance;


    Point fish = MakePoint(shoal.position_x[fishId], shoal.position_y[fishId]);
    Point newVelocity = MakePoint(shoal.velocity_x[fishId], shoal.velocity_y[fishId]);
    Point centerOfMass = MakePoint(0, 0);
    Point separation = MakePoint(0, 0);
    Point avrVelocity = MakePoint(0, 0);
    int neighboursCountCenterOfMass = 0;
    int neighboursCountVelocity = 0;

    int leftDownCornerCell = FishsCellId(MakePoint(fish.x - shoal.viewRange, fish.y - shoal.viewRange),
        grid.gridWidth, grid.gridHeight, grid.gridNumber_Horyzontal);
    int rightDownCornerCell = FishsCellId(MakePoint(fish.x + shoal.viewRange, fish.y - shoal.viewRange),
        grid.gridWidth, grid.gridHeight, grid.gridNumber_Horyzontal);
    int leftUpCornerCell = FishsCellId(MakePoint(fish.x - shoal.viewRange, fish.y + shoal.viewRange),
        grid.gridWidth, grid.gridHeight, grid.gridNumber_Horyzontal);

    int horyzontalCells = rightDownCornerCell - leftDownCornerCell;
    int verticalCells = (leftUpCornerCell - leftDownCornerCell) / grid.gridNumber_Horyzontal;

    for (int x = 0; x <= horyzontalCells; x++)
    {
        for (int y = 0; y <= verticalCells; y++)
        {
            int cellId = y * grid.gridNumber_Horyzontal + leftDownCornerCell + x;
            if (cellId > grid.gridNumber_Horyzontal * grid.gridNumber_Vertical - 1)
            {
                continue;
            }
            int start = grid.firstFishInCell[cellId];
            int end = grid.lastFishInCell[cellId];
            if (start == -1)
                continue;

            for (int neighbourFishId = start; neighbourFishId <= end; neighbourFishId++)
            {

                if (neighbourFishId == fishId)
                    continue;

                Point fishsFriend = MakePoint(shoal.position_x[neighbourFishId], shoal.position_y[neighbourFishId]);
                double powDistance = PowDistance(fish, fishsFriend);


                if (powViewRange > powDistance)
                {
                    neighboursCountCenterOfMass++;
                    centerOfMass.x += fishsFriend.x;
                    centerOfMass.y += fishsFriend.y;

                }

                if (powViewRange > powDistance)
                {
                    neighboursCountVelocity++;
                    avrVelocity.x += shoal.velocity_x[neighbourFishId];
                    avrVelocity.y += shoal.velocity_x[neighbourFishId];
                }

                if (powMinDistance > powDistance)
                {
                    separation.x -= fishsFriend.x - fish.x;
                    separation.y -= fishsFriend.y - fish.y;
                }
            }
        }
    }


    if (neighboursCountCenterOfMass > 0)
    {
        centerOfMass.x /= neighboursCountCenterOfMass;
        centerOfMass.y /= neighboursCountCenterOfMass;

        newVelocity.x += (centerOfMass.x - fish.x) * CohesionScale;
        newVelocity.y += (centerOfMass.y - fish.y) * CohesionScale;
    }



    separation.x *= SeparationScale;
    separation.y *= SeparationScale;

    newVelocity.x += separation.x;
    newVelocity.y += separation.y;

    if (neighboursCountVelocity != 0) {
        avrVelocity.x /= neighboursCountVelocity;
        avrVelocity.y /= neighboursCountVelocity;
        newVelocity.x += avrVelocity.x * AlignmentScale;
        newVelocity.y += avrVelocity.y * AlignmentScale;
    }

    // zmniejszam szybkość do maksymalnej jeśli jest to niezbędne
    if (PowDistance(MakePoint(0, 0), newVelocity) > MaxSpeed * MaxSpeed)
    {
        double calculateSpeed = sqrt(PowDistance(MakePoint(0, 0), newVelocity));
        newVelocity.x *= MaxSpeed / calculateSpeed;
        newVelocity.y *= MaxSpeed / calculateSpeed;
    }

    // odbicie od ściany
    if (fish.x + newVelocity.x < 0 || fish.x + newVelocity.x > width)
    {
        newVelocity.x = -newVelocity.x;
    }

    if (fish.y + newVelocity.y < 0 || fish.y + newVelocity.y > height)
    {
        newVelocity.y = -newVelocity.y;
    }

    // zapisanie wyników
    if (!(fish.x + newVelocity.x < 0 || fish.x + newVelocity.x > width))
    {
        fish.x += newVelocity.x;
        shoal.velocity_x[fishId] = newVelocity.x;
        shoal.position_x[fishId] = fish.x;
    }


    if (!(fish.y + newVelocity.y < 0 || fish.y + newVelocity.y > height))
    {
        fish.y += newVelocity.y;
        shoal.velocity_y[fishId] = newVelocity.y;
        shoal.position_y[fishId] = fish.y;
    }


    // zapisanie kordynatów rybki
    Point p1, p2, p3;
    FishToCordinates(fish, Direction(newVelocity), shoal.h, shoal.w, &p1, &p2, &p3);
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

    if (x >= fishNumber)
        return;

    int distance = shoal.minDistance;
    int startPoint_x = (width - distance * fisheGrideWidth) / 2;
    int startPoimt_y = (height - distance * fishGrideHeight) / 2;
    int fishGridY = x / fisheGrideWidth;
    int fishGridX = x - fishGridY * fisheGrideWidth;
    shoal.velocity_x[x] = MaxSpeed * cos(x * 2 * M_PI / fishNumber);
    shoal.velocity_y[x] = MaxSpeed * sin(x * 2 * M_PI / fishNumber);
    shoal.position_x[x] = startPoint_x + fishGridX * distance;
    shoal.position_y[x] = startPoimt_y + fishGridY * distance;
}

__global__ void ResetGridStartEnd(int* start, int* end, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;
    start[x] = -1;
    end[x] = -1;
}

__global__ void CalculateStartEnd(int* start, int* end, int* gridId)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= fishNumber)
        return;

    int curentGridId = gridId[x];

    if (x == 0 || gridId[x - 1] != curentGridId)
        start[curentGridId] = x;

    if (x == fishNumber - 1 || gridId[x + 1] != curentGridId)
        end[curentGridId] = x;

}





void CalculateNeededThreads(int* threads, int* blocks, int neededThreads)
{
    if (neededThreads < maxThreds)
    {
        *threads = neededThreads;
        *blocks = 1;
    }
    else
    {
        *threads = maxThreds;
        *blocks = (int)ceil((double)neededThreads / maxThreds);
        if (*blocks > maxBlocks)
            exit(-1);
    }
}

__global__ void ResetMapper(int* mapper, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;

    mapper[x] = x;
}

__global__ void MappShoal(Shoal shoal, Shoal tmpShoal, Grid grid, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;

    int mapFrom = grid.gridMapper[x];
    tmpShoal.velocity_x[x] = shoal.velocity_x[mapFrom];
    tmpShoal.velocity_y[x] = shoal.velocity_y[mapFrom];
    tmpShoal.position_x[x] = shoal.position_x[mapFrom];
    tmpShoal.position_y[x] = shoal.position_y[mapFrom];

}

__global__ void CopyShoal(Shoal shoal, Shoal tmpShoal, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;

    shoal.velocity_x[x] = tmpShoal.velocity_x[x];
    shoal.velocity_y[x] = tmpShoal.velocity_y[x];
    shoal.position_x[x] = tmpShoal.position_x[x];
    shoal.position_y[x] = tmpShoal.position_y[x];
}



void LunchCuda()
{
    int blocks, threads;
    CalculateNeededThreads(&threads, &blocks, fishNumber);
    CategorizeFishToCells << <blocks, threads >> > (shoal, grid);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    CalculateNeededThreads(&threads, &blocks, grid.gridNumber_Horyzontal * grid.gridNumber_Vertical);
    ResetGridStartEnd << <blocks, threads >> > (grid.firstFishInCell, grid.lastFishInCell, grid.gridWidth * grid.gridHeight - 1);
    CalculateNeededThreads(&threads, &blocks, fishNumber);
    ResetMapper << <blocks, threads >> > (grid.gridMapper, fishNumber);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::sort_by_key(thrust::device, grid.cellsId, grid.cellsId + fishNumber, grid.gridMapper);

    CalculateNeededThreads(&threads, &blocks, fishNumber);
    MappShoal << <blocks, threads >> > (shoal, scatterShoal, grid, fishNumber);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    CopyShoal << <blocks, threads >> > (shoal, scatterShoal, fishNumber);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    CalculateNeededThreads(&threads, &blocks, fishNumber);
    CalculateStartEnd << <blocks, threads >> > (grid.firstFishInCell, grid.lastFishInCell, grid.cellsId);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    CalculateNeededThreads(&threads, &blocks, fishNumber);
    CalculateShoal << <blocks, threads >> > (shoal, grid, device);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// Display callback function
void display() {
    cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
    glClear(GL_COLOR_BUFFER_BIT);

    LunchCuda();

    cudaGLUnmapBufferObject(buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glDrawArrays(GL_TRIANGLES, 0, 3 * fishNumber);
    glutSwapBuffers();

}

void InitCuda()
{
    gpuErrchk(cudaMalloc(&device, fishNumber * 6 * sizeof(float)));
    gpuErrchk(cudaMalloc(&shoal.position_x, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&shoal.position_y, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&shoal.velocity_x, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&shoal.velocity_y, fishNumber * sizeof(float)));

    gpuErrchk(cudaMalloc(&scatterShoal.position_x, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&scatterShoal.position_y, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&scatterShoal.velocity_x, fishNumber * sizeof(float)));
    gpuErrchk(cudaMalloc(&scatterShoal.velocity_y, fishNumber * sizeof(float)));

    gpuErrchk(cudaMalloc(&grid.cellsId, fishNumber * sizeof(int)));
    gpuErrchk(cudaMalloc(&grid.gridMapper, fishNumber * sizeof(int)));
    gpuErrchk(cudaMalloc(&grid.firstFishInCell, grid.gridHeight * grid.gridWidth * sizeof(int)));
    gpuErrchk(cudaMalloc(&grid.lastFishInCell, grid.gridHeight * grid.gridWidth * sizeof(int)));
    grid.gridNumber_Horyzontal = (int)ceil((double)width / (double)grid.gridWidth);
    grid.gridNumber_Vertical = (int)ceil((double)height / (double)grid.gridHeight);


    int rantagleOfFishWidth = (int)ceil(sqrt((fishNumber * width) / height));
    int rantagleOfFishHeight = (int)ceil(rantagleOfFishWidth * height / width);

    int blocks, threads;
    CalculateNeededThreads(&threads, &blocks, fishNumber);
    InitStartPosition << <blocks, threads >> > (shoal, rantagleOfFishWidth, rantagleOfFishHeight);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void time(int x)
{
    if (glutGetWindow())
    {
        FrameCounter();
        glutPostRedisplay();
        glutTimerFunc(Timer, time, 0);
    }
}

void Init()
{
    
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width, 0.0, height);
    glutDisplayFunc(display);
    time(0);
    glewInit();
    lastTime = glutGet(GLUT_ELAPSED_TIME);
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    unsigned int size = fishNumber * 6 * sizeof(float); // ilość wektorów
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    std::string vertexShader = readFileIntoString("vertexShader.glsl");
    std::string fragmentShader = readFileIntoString("fragmentShader.glsl");

    unsigned int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    InitCuda();
    cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
}

void FreeShoalOfFish()
{
    cudaFree(device);
    cudaFree(grid.lastFishInCell);
    cudaFree(grid.firstFishInCell);
    cudaFree(grid.cellsId);
    cudaFree(grid.gridMapper);

    cudaFree(shoal.velocity_x);
    cudaFree(shoal.velocity_y);
    cudaFree(shoal.velocity_x);
    cudaFree(shoal.velocity_y);

    cudaFree(scatterShoal.velocity_x);
    cudaFree(scatterShoal.velocity_y);
    cudaFree(scatterShoal.velocity_x);
    cudaFree(scatterShoal.velocity_y);
}

void FrameCounter()
{
    long long int  currentTime = glutGet(GLUT_ELAPSED_TIME);
    nbFrames++;
    if ((currentTime - lastTime)/1000 >= 1) { 
        char buffor[100];
        sprintf(buffor, "FPS: %d", nbFrames);
        glutSetWindowTitle(buffor);
        nbFrames = 0;
        lastTime = glutGet(GLUT_ELAPSED_TIME);
    }
}

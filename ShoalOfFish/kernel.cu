
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL/glut.h>

int x = 0;
float t = 0.0f;

__global__ void addKernel(int* c, const int* a, const int* b)
{
}

// Global variables to track the position of the dot
void Init()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    //glColor3f(0.0, 0.0, 0.0);
    //glPointSize(5);

    glMatrixMode(GL_PROJECTION);    //coordinate system
    //glLoadIdentity();
    gluOrtho2D(0.0, 1200.0, 0.0, 800.0);
}

void time(int x)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(10, time, 0);
        t += 0.0166f;
    }
}


// Display callback function
void display() {
    // Clear the window
    glClear(GL_COLOR_BUFFER_BIT); // clear display window

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double w = glutGet(GLUT_WINDOW_WIDTH);
    const double h = glutGet(GLUT_WINDOW_HEIGHT);
    gluOrtho2D(0.0, w, 0.0, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // Draw the dot at the current position
    glBegin(GL_POINTS);
    for (int i = 0; i < 300; i++)
    {
        glColor3f(0.3, 0.3, 0.3);
        glPointSize(5.0f);  // wat
        glVertex2i(x + i, h - 200);
    }
    glEnd();

    // Swap the buffers to display the dot
    glFlush();

    // Increment the x coordinate of the dot
    x+= 10;

    // If the dot has reached the right edge of the window, reset its position to the left edge
    if (x > w) {
        x = 0;
    }
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);                          //initialize toolkit
                                                    //Request double buffered true color window with Z-buffer
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);   //display mode
    glutInitWindowSize(1200, 800);
    glutCreateWindow("Moving Dot"); // Create the window
    Init();
    
    

    // Set the display callback function
    glutDisplayFunc(display);
    time(0);
    // Enter the main loop
    glutMainLoop();

    return 0;
}
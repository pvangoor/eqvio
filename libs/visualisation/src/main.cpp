#include "Plotter.h"

#include "eigen3/Eigen/Core"

#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace Eigen;

void startGlutWindow();
void renderPoints();

vector<Vector3d> pts;
thread plottingThread;

int main(int argc, char **argv) {
    cout << "Hello GLUT!" << endl;

    // plot some random points
    int n = 50;
    for (int i=0; i<n; ++i) {
        pts.emplace_back(3*Vector3d::Random()+ Vector3d(1,1,1));
    }

    vector<Vector3d> line(5*n);
    for (int i=0; i<5*n; ++i) {
        line[i] = Vector3d(cos(0.1*i),0.02*sqrt(i),sin(0.1*i));
    }

    Plotter p;
    p.hold = true;
    p.drawPoints(pts, Vector4d(0,0,1,0), 3);
    p.drawAxes(Matrix4d::Identity(), 1, 4);
    p.drawLine(line, Vector4d(0,0,0,1), 2);
    p.hold = false;


    p.maintain();

    // startGlutWindow();
    // plottingThread.join();

    return 0;

}

void startGlutWindow() {
    // Initialise glut
    char * my_argv[1];
    int my_argc = 1;
    my_argv[0] = strdup("Plotter");

    glutInit(&my_argc, my_argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Plotter window");

    // register callbacks
    glutDisplayFunc(renderPoints);
    glutIdleFunc(renderPoints);

    // Enter glut main loop
    plottingThread = thread(glutMainLoop);
}

void renderPoints() {
    glClearColor(0.1f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutSwapBuffers();
}
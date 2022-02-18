/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Plotter.h"
#include "GL/freeglut.h"
#include "eigen3/Eigen/Geometry"
#include <iostream>

using namespace std;
using namespace Eigen;

#define PI 3.14159265358979323846f /* pi */

PlotData3 makeXYGrid(const double& spacing = 1.0);
PlotData3 makeXYGrid(const double& spacing) {
    vector<Vector3d> dataPoints((11 + 11) * 2);

    for (int i = 0; i <= 10; ++i) {
        // X line
        dataPoints[4 * i + 0] = Vector3d((i - 5) * spacing, -5 * spacing, 0);
        dataPoints[4 * i + 1] = Vector3d((i - 5) * spacing, 5 * spacing, 0);
        // Y line
        dataPoints[4 * i + 2] = Vector3d(-5 * spacing, (i - 5) * spacing, 0);
        dataPoints[4 * i + 3] = Vector3d(5 * spacing, (i - 5) * spacing, 0);
    }

    PlotData3 gridData = PlotData3(dataPoints, GL_LINES, Vector4d(0, 0, 0, 0.5), 1);
    return gridData;
}

Plotter* Plotter::currentInstance;

Plotter::Plotter() {
    currentInstance = this;

    this->ratio = 1280 * 1.0 / 720;

    // Set up a grid
    this->savedData.emplace_back(makeXYGrid());

    // Start the plotting thread
    this->plottingThread = thread(currentInstance->startThread);
}

void Plotter::startThread() {
    // Initialise glut
    char* my_argv[1];
    int my_argc = 1;
    my_argv[0] = strdup("Plotter");

    glutInit(&my_argc, my_argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    if (glutGet(GLUT_SCREEN_HEIGHT) <= 1080) {
        glutInitWindowSize(1280, 720);
    } else {
        glutInitWindowSize(1920, 1080);
    }
    glutCreateWindow("Plotter window");

    // register callbacks
    glutDisplayFunc(&Plotter::wrap_renderPoints);
    glutReshapeFunc(&Plotter::wrap_changeWindowSize);
    glutIdleFunc(&Plotter::wrap_renderPoints);

    // respond to mouse
    glutMouseFunc(&Plotter::wrap_mouseButton);
    glutMotionFunc(&Plotter::wrap_mouseMove);

    // respond to keyboard
    glutKeyboardFunc(&Plotter::wrap_processNormalKeys);

    // Enter glut main loop
    glutMainLoop();
}

void Plotter::updatePlotData(const PlotData3& plotData) {
    if (this->pause)
        return;

    std::unique_lock lck(plottingMutex);
    if (!this->hold)
        this->plotsData = this->savedData;
    this->plotsData.emplace_back(plotData);
}

void Plotter::drawPoints(const vector<Vector3d>& newPoints, const Vector4d& color, const int& size) {
    this->updatePlotData(PlotData3(newPoints, GL_POINTS, color, size));
}
void Plotter::drawPoints(const vector<Vector3d>& newPoints, const vector<Vector4d>& colors, const int& size) {
    this->updatePlotData(PlotData3(newPoints, GL_POINTS, colors, size));
}

void Plotter::drawLine(const vector<Vector3d>& newLine, const Vector4d& color, const int& size) {
    this->updatePlotData(PlotData3(newLine, GL_LINE_STRIP, color, size));
}

void Plotter::drawAxes(const Matrix4d& pose, const double& length, const int& size) {
    Vector3d origin = pose.block<3, 1>(0, 3);

    PlotData3 xAxis({origin, origin + length * pose.block<3, 1>(0, 0)}, GL_LINES, Vector4d(1, 0, 0, 1), size);
    PlotData3 yAxis({origin, origin + length * pose.block<3, 1>(0, 1)}, GL_LINES, Vector4d(0, 1, 0, 1), size);
    PlotData3 zAxis({origin, origin + length * pose.block<3, 1>(0, 2)}, GL_LINES, Vector4d(0, 0, 1, 1), size);

    this->updatePlotData(xAxis);
    this->updatePlotData(yAxis);
    this->updatePlotData(zAxis);
}

void Plotter::changeWindowSize(int w, int h) {

    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if (h == 0)
        h = 1;

    this->ratio = w * 1.0 / h;

    // Use the Projection Matrix
    glMatrixMode(GL_PROJECTION);

    // Reset Matrix
    glLoadIdentity();

    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    // Set the correct perspective.
    gluPerspective(45, ratio, 1, 100);

    // Get Back to the Modelview
    glMatrixMode(GL_MODELVIEW);
}

void Plotter::renderPoints() {
    glClearColor(0.75f, 0.75f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, ratio, 1, 1e8);

    Vector3d eye = plotOrigin + zoom * Vector3d(
                                           cos(angleIncline) * cos(angleAzimuth), cos(angleIncline) * sin(angleAzimuth),
                                           sin(angleIncline));

    gluLookAt(
        eye.x(), eye.y(), eye.z(),                      // eye xyz
        plotOrigin.x(), plotOrigin.y(), plotOrigin.z(), // scene centre
        0.0f, 0.0f, 1.0f                                // 'Up' vector
    );

    // Draw things
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    {
        std::unique_lock lck(plottingMutex);
        for (const PlotData3& plot : plotsData) {
            plot.draw();
        }
    }

    glutSwapBuffers();
}

void Plotter::mouseButton(int button, int state, int x, int y) {
    // move when left mouse button is pressed.
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_UP) {
            this->mouseOriginX = -1;
            this->mouseOriginY = -1;
            this->plotMotionType = PlotMotion::NONE;
        } else {
            this->mouseOriginX = x;
            this->mouseOriginY = y;
            this->plotMotionType = PlotMotion::ROTATE;
        }
    } else if (button == GLUT_MIDDLE_BUTTON) {
        if (state == GLUT_UP) {
            this->mouseOriginX = -1;
            this->mouseOriginY = -1;
            this->plotMotionType = PlotMotion::NONE;
        } else {
            this->mouseOriginX = x;
            this->mouseOriginY = y;
            this->plotMotionType = PlotMotion::SCALE;
        }
    } else if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_UP) {
            this->mouseOriginX = -1;
            this->mouseOriginY = -1;
            this->plotMotionType = PlotMotion::NONE;
        } else {
            this->mouseOriginX = x;
            this->mouseOriginY = y;
            this->plotMotionType = PlotMotion::TRANSLATE;
        }
    }
}

void Plotter::mouseMove(int x, int y) {
    if (plotMotionType == PlotMotion::ROTATE) {
        if (this->mouseOriginX >= 0) {
            // update rotation
            this->angleAzimuth += -(x - this->mouseOriginX) * 0.01f;
            this->mouseOriginX = x;

            this->angleIncline =
                clamp(this->angleIncline + (y - this->mouseOriginY) * 0.01f, -PI / 2 - 1e-6f, PI / 2 - 1e-6f);
            this->mouseOriginY = y;
        }
    } else if (plotMotionType == PlotMotion::SCALE) {
        if (this->mouseOriginY >= 0) {
            this->zoom *= 1.0f + (y - this->mouseOriginY) * 0.01f;
            this->mouseOriginY = y;
        }
    } else if (plotMotionType == PlotMotion::TRANSLATE && allowTranslation) {
        if (this->mouseOriginX >= 0 && this->mouseOriginY >= 0) {
            float transX = -(x - this->mouseOriginX) * 0.001f;
            this->mouseOriginX = x;

            float transY = (y - this->mouseOriginY) * 0.001f;
            this->mouseOriginY = y;

            Vector3d trans = zoom * Vector3d(-sin(angleAzimuth) * transX, cos(angleAzimuth) * transX, transY);

            this->plotOrigin += trans;
        }
    }
}

void Plotter::processNormalKeys(unsigned char key, int x, int y) {
    if (key == 27) { // Esc
        exit(0);
    }
    if (key == ' ') { // space
        this->pause = !this->pause;
    }
}
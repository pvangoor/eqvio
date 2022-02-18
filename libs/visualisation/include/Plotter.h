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

#pragma once

#include "GL/glut.h"
#include <mutex>
#include <thread>

#include "eigen3/Eigen/Eigen"
#include <vector>

#include "PlotDataTypes.h"

enum class PlotMotion { NONE, ROTATE, TRANSLATE, SCALE };

class Plotter {
  protected:
    // Technical
    static Plotter* currentInstance;
    std::thread plottingThread;
    std::mutex plottingMutex;

    // Plot view interaction
    float ratio = 1;
    float angleAzimuth = 0;
    float angleIncline = 0;
    float zoom = 10.0;

    int mouseOriginX = -1;
    int mouseOriginY = -1;

    float deltaAngle = 0;

    PlotMotion plotMotionType = PlotMotion::NONE;

    Eigen::Vector3d plotOrigin = Eigen::Vector3d::Zero();
    bool allowTranslation = true;

    // Data
    std::vector<PlotData3> plotsData;
    std::vector<PlotData3> savedData;

  public:
    Plotter();
    ~Plotter() { plottingThread.join(); };

    void drawPoints(
        const std::vector<Eigen::Vector3d>& newPoints, const Eigen::Vector4d& color = Eigen::Vector4d(0, 0, 1, 1),
        const int& size = 1);
    void drawPoints(
        const std::vector<Eigen::Vector3d>& newPoints, const std::vector<Eigen::Vector4d>& color, const int& size = 1);
    void drawLine(
        const std::vector<Eigen::Vector3d>& newLine, const Eigen::Vector4d& color = Eigen::Vector4d(0, 0, 1, 1),
        const int& size = 1);
    void drawAxes(const Eigen::Matrix4d& pose, const double& length = 1.0, const int& size = 1);
    void updatePlotData(const PlotData3& plotData);

    bool hold = false;
    bool pause = false;

    void lockOrigin(const Eigen::Vector3d& origin = Eigen::Vector3d::Zero()) {
        allowTranslation = false;
        plotOrigin = origin;
    };
    void unlockOrigin() { allowTranslation = true; };

  protected:
    static void startThread();

    // Plotting functionality wrappers
    static void wrap_renderPoints() { currentInstance->renderPoints(); };
    static void wrap_changeWindowSize(int w, int h) { currentInstance->changeWindowSize(w, h); };
    static void wrap_mouseButton(int button, int state, int x, int y) {
        currentInstance->mouseButton(button, state, x, y);
    };
    static void wrap_processNormalKeys(unsigned char key, int x, int y) {
        currentInstance->processNormalKeys(key, x, y);
    };
    static void wrap_mouseMove(int x, int y) { currentInstance->mouseMove(x, y); };

    // Plotting functionality functions
    void changeWindowSize(int w, int h);
    void renderPoints();
    void mouseButton(int button, int state, int x, int y);
    void mouseMove(int x, int y);
    virtual void processNormalKeys(unsigned char key, int x, int y);

    // Utility
    template <class T> T clamp(const T& value, const T& minValue, const T& maxValue) {
        T result = value;
        result = (result < minValue) ? minValue : result;
        result = (result > maxValue) ? maxValue : result;
        return result;
    }
};
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

#include "eigen3/Eigen/Dense"
#include "eqvio/Geometry.h"
#include "eqvio/IMUVelocity.h"
#include "eqvio/VIOGroup.h"
#include "eqvio/VIOState.h"
#include "gtest/gtest.h"

GIFT::GICameraPtr createDefaultCamera();

VIOGroup randomGroupElement(const std::vector<int>& ids);
VIOGroup reasonableGroupElement(const std::vector<int>& ids);
VIOState randomStateElement(const std::vector<int>& ids);
VIOState reasonableStateElement(const std::vector<int>& ids);
IMUVelocity randomVelocityElement();
Eigen::VectorXd stateVecDiff(const VIOState& xi1, const VIOState& xi2);
double logNorm(const VIOGroup& X);

VisionMeasurement randomVisionMeasurement(const std::vector<int>& ids);
double stateDistance(const VIOState& xi1, const VIOState& xi2);
double measurementDistance(const VisionMeasurement& y1, const VisionMeasurement& y2);

void assertMatrixEquality(const Eigen::MatrixXd& M1, const Eigen::MatrixXd& M2, double h = -1.0);

template <typename F>
void testDifferential(const F& f, const Eigen::VectorXd& x, const Eigen::MatrixXd& Df, double h = -1.0) {
    // Check that each partial derivative is correct
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<double>::epsilon());
    }
    Eigen::MatrixXd numericalDf = numericalDifferential(f, x, h);
    assertMatrixEquality(Df, numericalDf, h);
}
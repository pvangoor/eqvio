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

#include <random>

#include "eigen3/Eigen/Dense"
#include "eqvio/mathematical/Geometry.h"
#include "eqvio/mathematical/IMUVelocity.h"
#include "eqvio/mathematical/VIOGroup.h"
#include "eqvio/mathematical/VIOState.h"
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

template <typename T, typename S>
std::vector<T> weightedResample(const std::vector<T>& samples, const std::vector<S>& weights) {
    static std::mt19937 rng = std::mt19937(0);
    static std::uniform_real_distribution<double> dist{0, 1};
    assert(samples.size() == weights.size());
    const int N = samples.size();
    std::vector<T> resamples(N);

    int j = 0;
    double totalWeight = weights.at(0);
    for (int k = 0; k < N; ++k) {
        const double nextThreshold = (dist(rng) + k) / N;
        while (totalWeight < nextThreshold && j + 1 < N) {
            ++j;
            totalWeight += weights.at(j);
        }
        resamples.at(k) = samples.at(j);
    }

    return resamples;
}
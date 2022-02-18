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
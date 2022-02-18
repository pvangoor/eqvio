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

#include "testing_utilities.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

VIOState reasonableStateElement(const std::vector<int>& ids) {
    VIOState xi;
    xi.sensor.inputBias.setRandom();
    xi.sensor.pose.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.pose.x.setRandom();
    xi.sensor.cameraOffset.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.cameraOffset.x.setRandom();
    xi.sensor.velocity.setRandom();

    xi.sensor.cameraOffset.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.cameraOffset.x.setRandom();

    xi.cameraLandmarks.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        xi.cameraLandmarks[i].p = Vector3d::Random() * 10;
        xi.cameraLandmarks[i].p.z() += 20.0;
        xi.cameraLandmarks[i].id = ids[i];
    }

    return xi;
}

VIOState randomStateElement(const vector<int>& ids) {
    VIOState xi;
    xi.sensor.inputBias.setRandom();
    xi.sensor.pose.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.pose.x.setRandom();
    xi.sensor.cameraOffset.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.cameraOffset.x.setRandom();
    xi.sensor.velocity.setRandom();

    xi.sensor.cameraOffset.R.fromQuaternion(Quaterniond::UnitRandom());
    xi.sensor.cameraOffset.x.setRandom();

    xi.cameraLandmarks.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        xi.cameraLandmarks[i].p = Vector3d::Random() * 10;
        xi.cameraLandmarks[i].id = ids[i];
    }

    return xi;
}

VectorXd stateVecDiff(const VIOState& xi1, const VIOState& xi2) {
    // Compute the tangent vector from xi1 to xi2
    VectorXd vecDiff = VectorXd(6 + 3 + 3 * xi1.cameraLandmarks.size());
    vecDiff.setConstant(NAN);
    vecDiff.block<6, 1>(0, 0) = SE3d::log(xi1.sensor.pose.inverse() * xi2.sensor.pose);
    vecDiff.block<3, 1>(6, 0) = (xi2.sensor.velocity - xi1.sensor.velocity);
    assert(xi1.cameraLandmarks.size() == xi2.cameraLandmarks.size());
    for (int i = 0; i < xi1.cameraLandmarks.size(); ++i) {
        assert(xi1.cameraLandmarks[i].id == xi2.cameraLandmarks[i].id);
        vecDiff.block<3, 1>(9 + 3 * i, 0) = (xi2.cameraLandmarks[i].p - xi1.cameraLandmarks[i].p);
    }
    assert(!vecDiff.hasNaN());
    return vecDiff;
}

IMUVelocity randomVelocityElement() {
    IMUVelocity vel;
    vel.gyr.setRandom();
    vel.acc.setRandom();
    vel.gyrBiasVel.setRandom();
    vel.accBiasVel.setRandom();
    vel.stamp = 0;
    return vel;
}

VIOGroup randomGroupElement(const vector<int>& ids) {
    VIOGroup X;
    X.beta.setRandom();
    X.A.R.fromQuaternion(Quaterniond::UnitRandom());
    X.A.x.setRandom();
    X.B.R.fromQuaternion(Quaterniond::UnitRandom());
    X.B.x.setRandom();
    X.w.setRandom();
    X.id = ids;
    X.Q.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        X.Q[i].R.fromQuaternion(Quaterniond::UnitRandom());
        X.Q[i].a = 2.0 * (double)rand() / RAND_MAX + 1.0;
    }

    return X;
}

VIOGroup reasonableGroupElement(const std::vector<int>& ids) {
    VIOGroup X;
    X.beta = Matrix<double, 6, 1>::Random() * 0.1;
    X.A = SE3d::exp(se3d::Random() * 0.1);
    X.B = SE3d::exp(se3d::Random() * 0.1);
    X.w = Vector3d::Random() * 0.1;
    X.id = ids;
    X.Q.resize(ids.size());
    for (int i = 0; i < ids.size(); ++i) {
        X.Q[i].R = SO3d::exp(Vector3d::Random() * 0.02);
        X.Q[i].a = 2.0 * (double)rand() / RAND_MAX + 1.0;
    }

    return X;
}

double logNorm(const VIOGroup& X) {
    double result = 0;
    result += SE3d::log(X.A).norm();
    result += SE3d::log(X.B).norm();
    result += X.w.norm();
    for (const SOT3d& Qi : X.Q) {
        result += SOT3d::log(Qi).norm();
    }
    return result;
}

double stateDistance(const VIOState& xi1, const VIOState& xi2) {
    double dist = 0;
    dist += (xi1.sensor.inputBias - xi2.sensor.inputBias).norm();
    dist += SE3d::log(xi1.sensor.pose.inverse() * xi2.sensor.pose).norm();
    dist += SE3d::log(xi1.sensor.cameraOffset.inverse() * xi2.sensor.cameraOffset).norm();
    dist += (xi1.sensor.velocity - xi2.sensor.velocity).norm();
    assert(xi1.cameraLandmarks.size() == xi2.cameraLandmarks.size());
    for (int i = 0; i < xi1.cameraLandmarks.size(); ++i) {
        assert(xi1.cameraLandmarks[i].id == xi2.cameraLandmarks[i].id);
        dist += (xi1.cameraLandmarks[i].p - xi2.cameraLandmarks[i].p).norm();
    }
    return dist;
}

VisionMeasurement randomVisionMeasurement(const vector<int>& ids) {
    VisionMeasurement result;
    result.cameraPtr = createDefaultCamera();
    result.stamp = 0.0;
    for (int i = 0; i < ids.size(); ++i) {
        Vector3d p;
        do {
            p.setRandom();
            p.normalize();
        } while (p.z() < 1e-1);
        result.camCoordinates[ids[i]] = result.cameraPtr->projectPoint(p);
    }
    return result;
}

double measurementDistance(const VisionMeasurement& y1, const VisionMeasurement& y2) {
    double dist = 0;
    assert(y1.camCoordinates.size() == y2.camCoordinates.size());
    double scale = max(VectorXd(y1).norm(), VectorXd(y2).norm());
    VectorXd diff = (y1 - y2);
    double result = diff.norm() / scale;
    return result;
}

GIFT::GICameraPtr createDefaultCamera() {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 450;
    K.at<double>(1, 1) = 450;
    K.at<double>(0, 2) = 400;
    K.at<double>(1, 2) = 240;
    GIFT::PinholeCamera cam = GIFT::PinholeCamera(cv::Size2i(800, 480), K);
    GIFT::GICameraPtr camPtr = std::make_shared<GIFT::PinholeCamera>(cam);
    return camPtr;
}

void assertMatrixEquality(const Eigen::MatrixXd& M1, const Eigen::MatrixXd& M2, double h) {
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<double>::epsilon());
    }
    const int& n = M1.rows();
    const int& m = M1.cols();
    Eigen::MatrixXi errorMat(n, m);

    ASSERT_EQ(M1.rows(), M2.rows());
    ASSERT_EQ(M1.cols(), M2.cols());

    EXPECT_FALSE(M1.hasNaN());
    EXPECT_FALSE(M2.hasNaN());
    // Check each entry
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            EXPECT_NEAR(M1(i, j), M2(i, j), std::max(h, 5e3 * h * abs(M1(i, j)))) << "Entry (" << i << ", " << j << ")";
            if (abs(M1(i, j) - M2(i, j)) > std::max(h, 5e3 * h * abs(M1(i, j)))) {
                errorMat(i, j) = 1;
            } else {
                errorMat(i, j) = 0;
            }
        }
    }
    if ((errorMat.array() == 1).any()) {
        std::cout << "Error matrix:\n" << errorMat << std::endl;
    }
}
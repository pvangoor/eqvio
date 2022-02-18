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

#include "eqvio/EqFMatrices.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

const Eigen::MatrixXd EqFCoordinateSuite::outputMatrixC(
    const VIOState& xi0, const VIOGroup& X, const VisionMeasurement& y, const bool useEquivariance) const {
    // Rows and their corresponding output components
    // [2i, 2i+2): Landmark measurement i

    // Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    const int M = xi0.cameraLandmarks.size();
    const vector<int> ids = y.getIds();
    const int N = ids.size();
    MatrixXd CStar = MatrixXd::Zero(2 * N, VIOSensorState::CompDim + Landmark::CompDim * M);

    const VisionMeasurement yHat = measureSystemState(stateGroupAction(X, xi0), y.cameraPtr);

    for (int i = 0; i < M; ++i) {
        const int& idNum = xi0.cameraLandmarks[i].id;
        const Vector3d& qi0 = xi0.cameraLandmarks[i].p;
        const auto it_y = find(ids.begin(), ids.end(), idNum);
        const auto it_Q = find(X.id.begin(), X.id.end(), idNum);
        assert(it_Q != X.id.end());
        const int k = distance(X.id.begin(), it_Q);
        if (it_y != ids.end()) {

            assert(*it_y == *it_Q);
            assert(X.id[k] == idNum);

            const int j = distance(ids.begin(), it_y);
            CStar.block<2, 3>(2 * j, VIOSensorState::CompDim + 3 * i) =
                useEquivariance ? outputMatrixCiStar(qi0, X.Q[k], y.cameraPtr, y.camCoordinates.at(idNum))
                                : outputMatrixCi(qi0, X.Q[k], y.cameraPtr);
        }
    }

    assert(!CStar.hasNaN());
    return CStar;
}

const Eigen::Matrix<double, 2, 3> EqFCoordinateSuite::outputMatrixCi(
    const Eigen::Vector3d& q0, const liepp::SOT3d& QHat, const GIFT::GICameraPtr& camPtr) const {
    const Vector3d qHat = QHat.inverse() * q0;
    const Vector2d yHat = camPtr->projectPoint(qHat);
    return outputMatrixCiStar(q0, QHat, camPtr, yHat);
}
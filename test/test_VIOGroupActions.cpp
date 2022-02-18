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

#include "eigen3/Eigen/Dense"
#include "eqvio/VIOGroup.h"
#include "eqvio/VIOState.h"
#include "eqvio/VisionMeasurement.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(VIOActionTest, StateAction) {
    srand(0);
    vector<int> ids = {0, 1, 2, 3, 4};
    const VIOGroup groupId = VIOGroup::Identity(ids);
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X1 = randomGroupElement(ids);
        const VIOGroup X2 = randomGroupElement(ids);

        const VIOState xi0 = randomStateElement(ids);

        // Check the distance function works
        const double dist00 = stateDistance(xi0, xi0);
        EXPECT_LE(dist00, NEAR_ZERO);

        // Check action identity
        const VIOState xi0_id = stateGroupAction(groupId, xi0);
        const double dist0id = stateDistance(xi0_id, xi0);
        EXPECT_LE(dist0id, NEAR_ZERO);

        // Check action compatibility
        const VIOState xi1 = stateGroupAction(X2, stateGroupAction(X1, xi0));
        const VIOState xi2 = stateGroupAction(X1 * X2, xi0);
        const double dist12 = stateDistance(xi1, xi2);
        EXPECT_LE(dist12, NEAR_ZERO);
    }
}

TEST(VIOActionTest, OutputAction) {
    srand(0);
    vector<int> ids = {0, 1, 2, 3, 4};
    const VIOGroup groupId = VIOGroup::Identity(ids);
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X1 = randomGroupElement(ids);
        const VIOGroup X2 = randomGroupElement(ids);

        VisionMeasurement y0 = randomVisionMeasurement(ids);

        // Check the distance function works
        const double dist00 = measurementDistance(y0, y0);
        EXPECT_LE(dist00, 1e-5);

        // Check action identity
        const VisionMeasurement y0_id = outputGroupAction(groupId, y0);
        const double dist0id = measurementDistance(y0_id, y0);
        EXPECT_LE(dist0id, 1e-5);

        // Check action compatibility
        const VisionMeasurement y1 = outputGroupAction(X2, outputGroupAction(X1, y0));
        const VisionMeasurement y2 = outputGroupAction(X1 * X2, y0);
        const double dist12 = measurementDistance(y1, y2);
        EXPECT_LE(dist12, 1e-5);
    }
}

TEST(VIOActionTest, OutputEquivariance) {
    srand(0);
    vector<int> ids = {5, 0, 1, 2, 3, 4};
    const GIFT::GICameraPtr camPtr = createDefaultCamera();
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X = randomGroupElement(ids);
        const VIOState xi0 = randomStateElement(ids);

        // Check the state/output equivariance
        const VisionMeasurement y1 = measureSystemState(stateGroupAction(X, xi0), camPtr);
        const VisionMeasurement y2 = outputGroupAction(X, measureSystemState(xi0, camPtr));
        const double dist12 = measurementDistance(y1, y2);
        EXPECT_LE(dist12, 1e-5);
    }
}

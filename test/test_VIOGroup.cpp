#include "eigen3/Eigen/Dense"
#include "eqvio/VIOGroup.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;

TEST(VIOGroupTest, BasicOperations) {
    vector<int> allIds = {0, 1, 2, 3, 4};
    const VIOGroup groupId = VIOGroup::Identity(allIds);
    for (int testRep = 0; testRep < TEST_REPS; ++testRep) {
        const VIOGroup X1 = randomGroupElement(allIds);
        const VIOGroup X2 = randomGroupElement(allIds);
        const VIOGroup X3 = randomGroupElement(allIds);

        // Check inverse
        const double inverseError1 = logNorm(X1.inverse() * X1);
        const double inverseError2 = logNorm(X1 * X1.inverse());
        EXPECT_LE(inverseError1, NEAR_ZERO);
        EXPECT_LE(inverseError2, NEAR_ZERO);

        // Check associativity
        const VIOGroup result12 = (X1 * X2) * X3;
        const VIOGroup result23 = X1 * (X2 * X3);
        const double assocError1 = logNorm(result12.inverse() * result23);
        const double assocError2 = logNorm(result23.inverse() * result12);
        const double assocError3 = logNorm(result12 * result23.inverse());
        const double assocError4 = logNorm(result23 * result12.inverse());
        EXPECT_LE(assocError1, NEAR_ZERO);
        EXPECT_LE(assocError2, NEAR_ZERO);
        EXPECT_LE(assocError3, NEAR_ZERO);
        EXPECT_LE(assocError4, NEAR_ZERO);

        // Check identity
        const double idError1 = logNorm(groupId);
        const double idError2 = logNorm((X1 * groupId) * X1.inverse());
        const double idError3 = logNorm(X1.inverse() * (groupId * X1));
        EXPECT_LE(idError1, NEAR_ZERO);
        EXPECT_LE(idError2, NEAR_ZERO);
        EXPECT_LE(idError3, NEAR_ZERO);
    }
}

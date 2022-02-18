/*
    This file is part of GIFT.

    GIFT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GIFT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GIFT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GIFT/ParameterGroup.h"
#include "GIFT/PatchFeatureTracker.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "GIFT/Visualisation.h"

#include <fstream>

using namespace Eigen;
using namespace cv;
using namespace std;
using namespace GIFT;

class PFTTest : public ::testing::Test {
  protected:
    PFTTest() {
        img0 = imread(String(TEST_DATA_DIR) + String("img0.png"));
        img1 = imread(String(TEST_DATA_DIR) + String("img1.png"));
    }

  public:
    Mat img0, img1;
    GIFT::PatchFeatureTracker<> pftTrans;
    GIFT::PatchFeatureTracker<Affine2Group> pftAffine;
};

static ftype maxBadFeaturesFrac = 0.2;

TEST_F(PFTTest, DetectAndTrackTranslation) {
    pftTrans.settings.maxFeatures = 50;
    pftTrans.settings.minimumFeatureDistance = 20;
    pftTrans.settings.minimumRelativeQuality = 0.01;
    pftTrans.settings.patchSize = Size(9, 9);
    pftTrans.settings.pyramidLevels = 4;

    pftTrans.detectFeatures(img0);
    vector<GIFT::Feature> features0 = pftTrans.outputFeatures();

    Point2f translationVec = Point2f(20, 10);
    const Mat translationMat = (Mat_<double>(2, 3) << 1, 0, translationVec.x, 0, 1, translationVec.y);
    Mat shiftedImg0;
    warpAffine(img0, shiftedImg0, translationMat, img0.size());

    pftTrans.trackFeatures(shiftedImg0);
    vector<GIFT::Feature> features1 = pftTrans.outputFeatures();

    // Check basic logic
    ASSERT_EQ(features0.size(), features1.size());
    for (int i = 0; i < features0.size(); ++i) {
        const GIFT::Feature& lmi0 = features0[i];
        const GIFT::Feature& lmi1 = features1[i];

        EXPECT_EQ(lmi0.idNumber, lmi1.idNumber);
        EXPECT_EQ(lmi0.lifetime, 0);
        EXPECT_EQ(lmi1.lifetime, 1);
    }

    // Check tracking success
    int badFeatures = 0;
    for (int i = 0; i < features0.size(); ++i) {
        const GIFT::Feature& lmi0 = features0[i];
        const GIFT::Feature& lmi1 = features1[i];

        Point2f coordinateError = (lmi0.camCoordinates + translationVec - lmi1.camCoordinates);
        float coordinateErrorNorm = pow(coordinateError.dot(coordinateError), 0.5);
        if (coordinateErrorNorm > 0.1) {
            ++badFeatures;
        }
    }
    EXPECT_LE(badFeatures, maxBadFeaturesFrac * features0.size());
}

TEST_F(PFTTest, DetectAndTrackAffine) {
    pftAffine.settings.maxFeatures = 50;
    pftAffine.settings.minimumFeatureDistance = 20;
    pftAffine.settings.minimumRelativeQuality = 0.01;
    pftAffine.settings.patchSize = Size(15, 15);
    pftAffine.settings.pyramidLevels = 4;

    pftAffine.detectFeatures(img0);
    vector<GIFT::Feature> features0 = pftAffine.outputFeatures();

    Point2f translationVec = Point2f(10, 10);
    const Mat translationMat = (Mat_<double>(2, 3) << 1, 0, translationVec.x, 0, 1, translationVec.y);
    Mat shiftedImg0;
    warpAffine(img0, shiftedImg0, translationMat, img0.size());

    pftAffine.trackFeatures(shiftedImg0);
    vector<GIFT::Feature> features1 = pftAffine.outputFeatures();

    // Check basic logic
    ASSERT_EQ(features0.size(), features1.size());
    for (int i = 0; i < features0.size(); ++i) {
        const GIFT::Feature& lmi0 = features0[i];
        const GIFT::Feature& lmi1 = features1[i];

        EXPECT_EQ(lmi0.idNumber, lmi1.idNumber);
        EXPECT_EQ(lmi0.lifetime, 0);
        EXPECT_EQ(lmi1.lifetime, 1);
    }

    // Check tracking success
    int badFeatures = 0;
    for (int i = 0; i < features0.size(); ++i) {
        const GIFT::Feature& lmi0 = features0[i];
        const GIFT::Feature& lmi1 = features1[i];

        Point2f coordinateError = (lmi0.camCoordinates + translationVec - lmi1.camCoordinates);
        float coordinateErrorNorm = pow(coordinateError.dot(coordinateError), 0.5);
        if (coordinateErrorNorm > 0.1) {
            ++badFeatures;
        }
    }

    EXPECT_LE(badFeatures, maxBadFeaturesFrac * features0.size());
}
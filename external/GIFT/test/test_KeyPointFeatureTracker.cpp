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

#include "GIFT/KeyPointFeatureTracker.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

#include "GIFT/Visualisation.h"

#include <fstream>

using namespace std;
using namespace cv;
class KPFTTest : public ::testing::Test {
  protected:
    KPFTTest() {
        img0 = cv::imread(String(TEST_DATA_DIR) + String("img0.png"));
        img1 = cv::imread(String(TEST_DATA_DIR) + String("img1.png"));
    }

  public:
    Mat img0, img1;
    GIFT::KeyPointFeatureTracker kpt;
};

TEST_F(KPFTTest, DetectAndTrack) {
    kpt.settings.maxFeatures = 100;
    kpt.settings.minimumFeatureDistance = 10;

    kpt.detectFeatures(img0);
    vector<GIFT::Feature> features0 = kpt.outputFeatures();

    Point2f translationVec = Point2f(1, 1);
    const Mat translationMat = (Mat_<double>(2, 3) << 1, 0, translationVec.x, 0, 1, translationVec.y);
    Mat shiftedImg0;
    warpAffine(img0, shiftedImg0, translationMat, img0.size());

    kpt.trackFeatures(shiftedImg0);
    vector<GIFT::Feature> features1 = kpt.outputFeatures();

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
    int bad_tracking = 0;
    for (int i = 0; i < features0.size(); ++i) {
        const GIFT::Feature& lmi0 = features0[i];
        const GIFT::Feature& lmi1 = features1[i];

        Point2f coordinateError = (lmi0.camCoordinates + translationVec - lmi1.camCoordinates);
        float coordinateErrorNorm = pow(coordinateError.dot(coordinateError), 0.5);

        if (coordinateErrorNorm > 2.0) {
            ++bad_tracking;
        }
    }

    EXPECT_LE(bad_tracking, 0.7 * features0.size());
}

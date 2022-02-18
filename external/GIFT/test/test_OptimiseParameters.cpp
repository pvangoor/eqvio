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

#include "GIFT/OptimiseParameters.h"
#include "GIFT/ParameterGroup.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace cv;
using namespace std;
using namespace GIFT;

class OptimiseParametersTest : public ::testing::Test {
  protected:
    OptimiseParametersTest() {
        img0 = imread(dataDir + String("img0.png"));
        cvtColor(img0, img0, COLOR_BGR2GRAY);

        img1 = imread(dataDir + String("img1.png"));
        cvtColor(img1, img1, COLOR_BGR2GRAY);

        const Point2f basePoint = Point2f(485, 155);
        const Point2f edgePoint = Point2f(350, 478);
        int numLevels = 4;

        img0GradientPyrBase = ImageWithGradientPyramid(img0, 1);
        img0ImagePyrBase = ImagePyramid(img0, 1);
        img0PatchBase = extractPyramidPatch(basePoint, Size(21, 21), img0GradientPyrBase);
        img0PatchEdgeBase = extractPyramidPatch(edgePoint, Size(10, 10), img0GradientPyrBase);

        img0GradientPyrLevels = ImageWithGradientPyramid(img0, numLevels);
        img0ImagePyrLevels = ImagePyramid(img0, numLevels);
        img0PatchLevels = extractPyramidPatch(basePoint, Size(21, 21), img0GradientPyrLevels);
        img0PatchEdgeLevels = extractPyramidPatch(edgePoint, Size(21, 21), img0GradientPyrBase);
    }

  public:
    String dataDir = String(TEST_DATA_DIR);
    Mat img0, img1;
    ImageWithGradientPyramid img0GradientPyrBase, img0GradientPyrLevels;
    ImagePyramid img0ImagePyrBase, img0ImagePyrLevels;
    PyramidPatch img0PatchBase, img0PatchLevels, img0PatchEdgeBase, img0PatchEdgeLevels;
};

TEST_F(OptimiseParametersTest, GetSubPixel) {
    // Check integer values match up
    for (int testIter = 0; testIter < 100; ++testIter) {
        Vector2i point = Vector2i::Random();
        point.x() = clamp(point.x(), 0, img0.cols - 1);
        point.y() = clamp(point.y(), 0, img0.rows - 1);
        Vector2T pointT(point.x(), point.y());

        float value = getSubPixel(img0, pointT);

        int trueValue = img0.at<uchar>(Point2i(point.x(), point.y()));

        EXPECT_FLOAT_EQ(value, trueValue);
    }

    // Check the extrapolation matches the patch
    // cout << img0PatchEdgeBase.vecImage[0] << endl;
    const Vector2T offset = 0.5 * Vector2T(img0PatchEdgeBase.rows() - 1, img0PatchEdgeBase.cols() - 1);
    for (int y = 0; y < img0PatchEdgeBase.rows(); ++y) {
        for (int x = 0; x < img0PatchEdgeBase.cols(); ++x) {
            Vector2T point = Vector2T(x, y) + img0PatchEdgeBase.centre() - offset;
            float value = getSubPixel(img0, point);
            EXPECT_FLOAT_EQ(value, img0PatchEdgeBase.at(y, x));
        }
    }
}

TEST_F(OptimiseParametersTest, TranslationAcceptsMinimumOnBase) {
    TranslationGroup params = TranslationGroup::Identity();
    optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

    ftype tsError = (params.translation).norm();
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, AffineAcceptsMinimumOnBase) {
    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

    ftype tfError = (params.transformation - Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();

    EXPECT_LE(tfError, 1e-3);
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, TranslationConvergeSmallErrorOnBase) {
    for (int testIter = 0; testIter < 10; ++testIter) {
        TranslationGroup params;
        params.translation = Vector2T::Random() * 3.0;

        optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

        ftype tsError = (params.translation).norm();
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, AffineConvergeSmallErrorOnBase) {
    for (int testIter = 0; testIter < 10; ++testIter) {
        Affine2Group params = Affine2Group::Identity();
        params.translation = Vector2T::Random() * 2.0;

        optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

        ftype tfError = (params.transformation - Matrix2T::Identity()).norm();
        ftype tsError = (params.translation).norm();

        EXPECT_LE(tfError, 1e-2);
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, TranslationAcceptsMinimumInLevels) {
    TranslationGroup params = TranslationGroup::Identity();
    optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

    ftype tsError = (params.translation).norm();
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, AffineAcceptsMinimumInLevels) {
    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

    ftype tfError = (params.transformation - Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();
    EXPECT_LE(tfError, 1e-2);
    EXPECT_LE(tsError, 1e-2);
}

TEST_F(OptimiseParametersTest, TranslationConvergeErrorInLevels) {
    for (int testIter = 0; testIter < 10; ++testIter) {
        TranslationGroup params;
        params.translation = Vector2T::Random() * 30.0;

        optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

        ftype tsError = (params.translation).norm();
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, AffineConvergeErrorInLevels) {
    for (int testIter = 0; testIter < 10; ++testIter) {
        Affine2Group params = Affine2Group::Identity();
        params.translation = Vector2T::Random() * 30.0;

        optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

        ftype tfError = (params.transformation - Matrix2T::Identity()).norm();
        ftype tsError = (params.translation).norm();
        EXPECT_LE(tfError, 1e-2);
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, TranslationAtEdge) {
    TranslationGroup params = TranslationGroup::Identity();
    params.translation.x() = 2;
    optimiseParameters(params, img0PatchEdgeBase, img0ImagePyrBase);

    ftype tsError = (params.translation).norm();
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, ManyPoints) {
    vector<Point2f> points;
    goodFeaturesToTrack(img0, points, 50, 0.05, 10);

    vector<PyramidPatch> patches = extractPyramidPatches(points, img0, Size(21, 21), 3);
    ImagePyramid pyr1(img1, 3);
    vector<Affine2Group> params(patches.size());
    for (int i = 0; i < params.size(); ++i) {
        params[i] = Affine2Group::Identity();
        optimiseParameters(params[i], patches[i], pyr1);
    }

    // Note this is slower than openCV methods, but they actually only implement translational tracking.
    // OpenCV does not consider (affine) warping of the patch.
}
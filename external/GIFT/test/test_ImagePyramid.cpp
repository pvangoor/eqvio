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

#include "GIFT/ImagePyramid.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "gtest/gtest.h"

using namespace GIFT;

TEST(ImagePyramidTest, PyrDimensions) {
    cv::Mat baseImage = cv::Mat::zeros(cv::Size(pow(2, 10), pow(2, 10)), CV_8UC1);
    ImagePyramid pyramid(baseImage, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(pyramid.levels[i].rows, baseImage.rows / pow(2, i));
        EXPECT_EQ(pyramid.levels[i].cols, baseImage.cols / pow(2, i));
    }
}

TEST(ImagePyramidTest, PatchExtraction) {
    cv::Mat img0 = cv::imread(cv::String(TEST_DATA_DIR) + cv::String("img0.png"));
    if (img0.channels() > 1) {
        cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
    }
    const cv::Size sze{10, 10};
    const cv::Point2f pt = cv::Point2f(img0.cols, img0.rows) / 2.0;

    cv::Mat patch0, patch1;
    cv::getRectSubPix(img0, sze, pt, patch0, CV_32F);
    patch1 = getPatchSubPix(sze, pt, img0, Eigen::Matrix2d::Identity());
    Eigen::VectorXd vec0 = vectoriseImage(patch0);
    Eigen::VectorXd vec1 = vectoriseImage(patch1);

    Eigen::VectorXd residual = vec1 - vec0;
    EXPECT_NEAR(residual.norm(), 0.0, 1e-6);
}
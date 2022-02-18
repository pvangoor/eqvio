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

#pragma once

#include "GIFT/Feature.h"
#include "opencv2/core.hpp"
#include <vector>

namespace GIFT {

cv::Mat drawFeatureImage(const cv::Mat& baseImage, const std::vector<Feature>& features, const int& radius = 3,
    const cv::Scalar& color = cv::Scalar(0, 255, 255));
cv::Mat drawFlowImage(const cv::Mat& baseImage, const std::vector<Feature>& features0,
    const std::vector<Feature>& features1, const int& radius = 3,
    const cv::Scalar& circleColor = cv::Scalar(0, 255, 255), const int& thickness = 2,
    const cv::Scalar& lineColor = cv::Scalar(255, 0, 255));
cv::Mat drawFlowImage(const cv::Mat& image0, const cv::Mat& image1, const std::vector<Feature>& features0,
    const std::vector<Feature>& features1, const int& radius = 3,
    const cv::Scalar& circleColor = cv::Scalar(0, 255, 255), const int& thickness = 2,
    const cv::Scalar& lineColor = cv::Scalar(0, 255, 0));

} // namespace GIFT
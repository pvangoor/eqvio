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
#include "GIFT/GIFeatureTracker.h"

using namespace GIFT;

// Initialisation and configuration
GIFeatureTracker::GIFeatureTracker(const std::shared_ptr<const GICamera> cameraParams, const cv::Mat& mask) {
    this->setCamera(cameraParams);
    this->setMask(mask);
}

GIFeatureTracker::GIFeatureTracker(const std::shared_ptr<const GICamera> cameraParams) {
    this->setCamera(cameraParams);
}

void GIFeatureTracker::setCamera(const std::shared_ptr<const GICamera> cameraParameters) {
    // cameraPtr = std::shared_ptr<const GICamera>(cameraParameters);
    // cameraPtr = std::make_shared<const GICamera>(cameraParameters);
    // cameraPtr.reset(cameraParameters);
    cameraPtr = cameraParameters;
}

void GIFeatureTracker::setMask(const cv::Mat& mask) { this->mask = mask; }

void GIFeatureTracker::processImage(const cv::Mat& image) {
    trackFeatures(image);
    detectFeatures(image);
}

void GIFeatureTracker::Settings::configure(const YAML::Node& node) {
    safeConfig(node["featureSearchThreshold"], featureSearchThreshold);
    safeConfig(node["maxFeatures"], maxFeatures);
}
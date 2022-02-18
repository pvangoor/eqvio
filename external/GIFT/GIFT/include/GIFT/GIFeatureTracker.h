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

#include <vector>

#include "GIFT/Feature.h"
#include "GIFT/camera/camera.h"
#include "opencv2/core.hpp"
#include "yaml-cpp/yaml.h"

#include <type_traits>

namespace GIFT {

class GIFeatureTracker {
  protected:
    int currentNumber = 0;
    std::shared_ptr<const GICamera> cameraPtr;
    cv::Mat mask;

  public:
    // Initialisation and configuration
    GIFeatureTracker(){};
    template <class CamClass, std::enable_if_t<std::is_base_of<GICamera, CamClass>::value, bool> = true>
    GIFeatureTracker(const CamClass& cameraParams) : GIFeatureTracker(std::make_shared<const CamClass>(cameraParams)) {}
    GIFeatureTracker(const std::shared_ptr<const GICamera> cameraParams);
    GIFeatureTracker(const std::shared_ptr<const GICamera> cameraParams, const cv::Mat& mask);
    virtual ~GIFeatureTracker(){};
    virtual void setCamera(const std::shared_ptr<const GICamera> cameraParameters);
    virtual void setMask(const cv::Mat& mask);

    struct Settings {
        ftype featureSearchThreshold = 0.8;
        int maxFeatures = 50;
        virtual void configure(const YAML::Node& node);
    };
    Settings settings;

    // Core
    virtual void processImage(const cv::Mat& image);
    virtual void detectFeatures(const cv::Mat& image) = 0;
    virtual void trackFeatures(const cv::Mat& image) = 0;
    virtual std::vector<Feature> outputFeatures() const = 0;
};
} // namespace GIFT

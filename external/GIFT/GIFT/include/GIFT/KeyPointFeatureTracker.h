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

#include "GIFT/GIFeatureTracker.h"

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

namespace GIFT {

class KeyPointFeatureTracker : public GIFeatureTracker {
  protected:
    struct InternalKPFeature {
        cv::KeyPoint kp;
        cv::Mat descriptor;
        int id = -1;
        int lifetime = 0;
        double descriptorDist = 0;
        cv::Point2f camCoordinates() const { return kp.pt; }
    };

    std::vector<InternalKPFeature> features; // Feature storage
    cv::Ptr<cv::ORB> ORBDetector = cv::ORB::create();
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

  public:
    // Settings
    struct Settings : GIFeatureTracker::Settings {
        double minimumFeatureDistance = 20;
    };
    Settings settings; // TODO expand these settings to actually change the detector

    // Initialisation and configuration
    KeyPointFeatureTracker() = default;
    KeyPointFeatureTracker(const std::shared_ptr<const GICamera> cameraParams) : GIFeatureTracker(cameraParams){};
    KeyPointFeatureTracker(const std::shared_ptr<const GICamera> cameraParams, const cv::Mat& mask)
        : GIFeatureTracker(cameraParams, mask){};
    template <class CamClass, std::enable_if_t<std::is_base_of<GICamera, CamClass>::value, bool> = true>
    KeyPointFeatureTracker(const CamClass& cameraParams) : GIFeatureTracker(cameraParams){};

    // Core
    virtual void detectFeatures(const cv::Mat& image) override;

    virtual void trackFeatures(const cv::Mat& image) override;

    [[nodiscard]] virtual std::vector<Feature> outputFeatures() const override;

    [[nodiscard]] Feature exportFeature(const InternalKPFeature& feature) const;

    void removePointsTooCloseToFeatures(std::vector<InternalKPFeature>& newKeypoints) const;
    static void filterForBestPoints(
        std::vector<InternalKPFeature>& proposedFeatures, const int& maxFeatures, const double& minDist);
};

} // namespace GIFT

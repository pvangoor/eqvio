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
#include "GIFT/ImagePyramid.h"
#include "GIFT/OptimiseParameters.h"
#include "GIFT/ParameterGroup.h"

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

namespace GIFT {

template <class PG = TranslationGroup, std::enable_if_t<std::is_base_of<ParameterGroup, PG>::value, bool> = true>
class PatchFeatureTracker : public GIFeatureTracker {
  protected:
    // Transform parameters and patches
    struct InternalPatchFeature {
        PyramidPatch patch;
        PG parameters;
        int id = -1;
        int lifetime = 0;
        cv::Point2f camCoordinates() const {
            const Eigen::Vector2T result = patch.centre() + parameters.applyLeftAction(Eigen::Vector2T(0, 0));
            return cv::Point2f(result.x(), result.y());
        }
    };

    std::vector<InternalPatchFeature> features; // Feature storage

  public:
    // Settings
    struct Settings : GIFeatureTracker::Settings {
        double minimumFeatureDistance = 20;
        double minimumRelativeQuality = 0.05;
        int pyramidLevels = 3;
        cv::Size patchSize = cv::Size(21, 21);
    };
    Settings settings;

    // Initialisation and configuration
    PatchFeatureTracker() = default;
    PatchFeatureTracker(const std::shared_ptr<const GICamera> cameraParams) : GIFeatureTracker(cameraParams){};
    PatchFeatureTracker(const std::shared_ptr<const GICamera> cameraParams, const cv::Mat& mask)
        : GIFeatureTracker(cameraParams, mask){};
    template <class CamClass, std::enable_if_t<std::is_base_of<GICamera, CamClass>::value, bool> = true>
    PatchFeatureTracker(const CamClass& cameraParams) : GIFeatureTracker(cameraParams){};

    // Core
    virtual void detectFeatures(const cv::Mat& image) override {
        // Detect new points
        std::vector<cv::Point2f> newPoints;
        cv::Mat gray = image;
        if (gray.channels() > 1)
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        goodFeaturesToTrack(
            gray, newPoints, settings.maxFeatures, settings.minimumRelativeQuality, settings.minimumFeatureDistance);

        // Remove new points that are too close to existing features
        std::vector<cv::Point2f> oldPoints(features.size());
        transform(features.begin(), features.end(), oldPoints.begin(),
            [](const InternalPatchFeature& f) { return f.camCoordinates(); });
        removePointsTooClose(newPoints, oldPoints, settings.minimumFeatureDistance);
        const int numPointsToAdd = settings.maxFeatures - oldPoints.size();
        newPoints.resize(std::max(numPointsToAdd, 0));

        // Convert the new points to patch features
        std::vector<PyramidPatch> newPatches =
            extractPyramidPatches(newPoints, gray, settings.patchSize, settings.pyramidLevels);
        auto newFeatureLambda = [this](const PyramidPatch& patch) {
            InternalPatchFeature feature;
            feature.patch = patch;
            feature.parameters = PG::Identity();
            feature.id = ++this->currentNumber;
            return feature;
        };
        std::vector<InternalPatchFeature> newFeatures(newPatches.size());
        transform(newPatches.begin(), newPatches.end(), newFeatures.begin(), newFeatureLambda);

        // Append
        features.insert(features.end(), newFeatures.begin(), newFeatures.end());

        // TODO: This only adds new features. Ideally, we also refresh existing features.
    };

    virtual void trackFeatures(const cv::Mat& image) override {
        cv::Mat gray = image;
        if (gray.channels() > 1)
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        ImagePyramid newPyr(gray, settings.pyramidLevels);
        std::for_each(features.begin(), features.end(), [&newPyr](InternalPatchFeature& feature) {
            optimiseParameters(feature.parameters, feature.patch, newPyr);
            ++feature.lifetime;
        });
        // TODO: We need to remove features that are no longer visible.
    };

    [[nodiscard]] virtual std::vector<Feature> outputFeatures() const override {
        std::vector<Feature> featuresOut(features.size());
        transform(features.begin(), features.end(), featuresOut.begin(),
            [this](const InternalPatchFeature& f) { return this->exportFeature(f); });
        return featuresOut;
    };

    [[nodiscard]] Feature exportFeature(const InternalPatchFeature& feature) const {
        Feature lm;
        lm.camCoordinates = feature.camCoordinates();
        lm.cameraPtr = cameraPtr;
        lm.idNumber = feature.id;
        lm.lifetime = feature.lifetime;
        return lm;
        // TODO: Some parts of the feature are missing. Is this a problem?
    }

    static void removePointsTooClose(
        std::vector<cv::Point2f> newPoints, const std::vector<cv::Point2f>& oldPoints, const double& minDist) {
        const double minDistSq = minDist * minDist;
        for (int i = newPoints.size() - 1; i >= 0; --i) {
            cv::Point2f& newPoint = newPoints[i];
            for (const cv::Point2f& point : oldPoints) {
                const double distSq =
                    (point.x - newPoint.x) * (point.x - newPoint.x) + (point.y - newPoint.y) * (point.y - newPoint.y);
                if (distSq < minDistSq) {
                    newPoints.erase(newPoints.begin() + i);
                    break;
                }
            }
        }
    }
};
} // namespace GIFT

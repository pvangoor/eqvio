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

#include <random>
#include <vector>

#include "yaml-cpp/yaml.h"
#include <GIFT/Feature.h>

namespace GIFT {

struct RansacParameters {
    size_t maxIterations = 5;
    size_t minDataPoints = 10;
    ftype inlierThreshold = 0.1;
    size_t minInliers = 20;
};

std::vector<GIFT::Feature> determineStaticWorldInliers(
    const std::vector<GIFT::Feature>& features, const RansacParameters& params, std::mt19937& generator);

Eigen::Matrix3T fitEssentialMatrix(const std::vector<GIFT::Feature>& features);

template <typename T> std::vector<T> sampleVector(const std::vector<T>& items, size_t n, std::mt19937& generator) {
    // This is basic reservoir sampling
    n = std::min(items.size(), n);
    std::vector<T> sample;
    sample.insert(sample.end(), items.begin(), items.begin() + n);

    for (size_t i = n; i < items.size(); ++i) {
        std::uniform_int_distribution<size_t> distribution(0, i);
        const std::size_t j = distribution(generator);
        if (j < sample.size()) {
            sample[j] = items[i];
        }
    }

    return sample;
}

}; // namespace GIFT

namespace YAML {
template <> struct convert<GIFT::RansacParameters> {
    static Node encode(const GIFT::RansacParameters& rhs) {
        Node node;
        node["maxIterations"] = rhs.maxIterations;
        node["minDataPoints"] = rhs.minDataPoints;
        node["inlierThreshold"] = rhs.inlierThreshold;
        node["minInliers"] = rhs.minInliers;
        return node;
    }

    static bool decode(const Node& node, GIFT::RansacParameters& rhs) {
        if (!node.IsMap()) {
            return false;
        }
        if (!(node["maxIterations"] && node["minDataPoints"] && node["inlierThreshold"] && node["minInliers"]))
            return false;

        rhs.maxIterations = node["maxIterations"].as<size_t>();
        rhs.minDataPoints = node["minDataPoints"].as<size_t>();
        rhs.inlierThreshold = node["inlierThreshold"].as<ftype>();
        rhs.minInliers = node["minInliers"].as<size_t>();
        return true;
    }
};
} // namespace YAML
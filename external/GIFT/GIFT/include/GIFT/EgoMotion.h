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
#include "eigen3/Eigen/Dense"

namespace GIFT {

class EgoMotion {
  public:
    Eigen::Vector3T linearVelocity;
    Eigen::Vector3T angularVelocity;
    ftype optimisedResidual = INFINITY;
    int optimisationSteps;
    int numberOfFeatures;

    static constexpr ftype optimisationThreshold = 1e-8;
    static constexpr int maxIterations = 30;

    EgoMotion(const std::vector<GIFT::Feature>& features, const ftype& dt = 1);
    EgoMotion(const std::vector<GIFT::Feature>& features, const Eigen::Vector3T& initLinVel, const ftype& dt = 1);
    EgoMotion(const std::vector<GIFT::Feature>& features, const Eigen::Vector3T& initLinVel,
        const Eigen::Vector3T& initAngVel, const ftype& dt = 1);
    EgoMotion(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& sphereFlows);
    EgoMotion(
        const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& sphereFlows, const Eigen::Vector3T& initLinVel);
    EgoMotion(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& sphereFlows,
        const Eigen::Vector3T& initLinVel, const Eigen::Vector3T& initAngVel);
    std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>> estimateFlows(
        const std::vector<GIFT::Feature>& features) const;
    std::vector<std::pair<cv::Point2f, Eigen::Vector2T>> estimateFlowsNorm(
        const std::vector<GIFT::Feature>& features) const;
    static Eigen::Vector3T estimateAngularVelocity(
        const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& sphereFlows,
        const Eigen::Vector3T& linVel = Eigen::Vector3T::Zero());

  private:
    static Eigen::Vector3T angularFromLinearVelocity(
        const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& flows, Eigen::Vector3T& linVel);
    static std::pair<int, ftype> optimize(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& flows,
        Eigen::Vector3T& linVel, Eigen::Vector3T& angVel);
    static void optimizationStep(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& flows,
        Eigen::Vector3T& linVel, Eigen::Vector3T& angVel);
    static ftype computeResidual(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& flows,
        const Eigen::Vector3T& linVel, const Eigen::Vector3T& angVel);
    static bool voteForLinVelInversion(const std::vector<std::pair<Eigen::Vector3T, Eigen::Vector3T>>& flows,
        const Eigen::Vector3T& linVel, const Eigen::Vector3T& angVel);
};

} // namespace GIFT
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

#include "GIFT/RANSAC.h"
#include <algorithm>
#include <limits>

std::vector<GIFT::Feature> GIFT::determineStaticWorldInliers(
    const std::vector<GIFT::Feature>& features, const GIFT::RansacParameters& params, std::mt19937& generator) {
    std::vector<GIFT::Feature> bestFitInliers(0);
    if (features.size() < params.minInliers || features.size() < params.minDataPoints || params.maxIterations == 0) {
        return features;
    }

    for (size_t iter = 0; iter < params.maxIterations; ++iter) {
        std::vector<GIFT::Feature> sampledFeatures = sampleVector(features, params.minDataPoints, generator);
        Eigen::Matrix3T sampleEssentialMatrix = GIFT::fitEssentialMatrix(sampledFeatures);

        // Determine inliers
        std::vector<GIFT::Feature> modelInliers(features.size());
        const auto endIt = std::copy_if(features.begin(), features.end(), modelInliers.begin(),
            [&params, &sampleEssentialMatrix](const GIFT::Feature& f) {
                const cv::Point2f& p2CV = f.camCoordinatesNorm();
                const Eigen::Vector3T& p2 = Eigen::Vector3T(p2CV.x, p2CV.y, 1.0);
                const Eigen::Vector3T& p1 = p2 - Eigen::Vector3T(f.opticalFlowNorm.x(), f.opticalFlowNorm.y(), 0.0);

                const ftype error = p1.transpose() * sampleEssentialMatrix * p2;
                return (error < params.inlierThreshold);
            });
        modelInliers.resize(std::distance(modelInliers.begin(), endIt));

        // The best model is determined by the highest number of inliers
        if (modelInliers.size() > bestFitInliers.size() && modelInliers.size() > params.minInliers) {
            bestFitInliers = modelInliers;
        }
    }

    return bestFitInliers;
}

Eigen::Matrix3T GIFT::fitEssentialMatrix(const std::vector<GIFT::Feature>& features) {
    // This uses the 8-point algorithm.
    // The resulting essential matrix E is expected to satisfy
    // p_{t-1}^\top E p_t = 0, for every feature point p.

    // Populate the matrix A that satisfies A f = 0, where f are the vectorised matrix entries of the solution.
    Eigen::MatrixXT A(features.size(), 9);
    for (size_t i = 0; i < features.size(); ++i) {
        const GIFT::Feature& f = features[i];
        const cv::Point2f& p2 = f.camCoordinatesNorm();
        const cv::Point2f& p1 = p2 - cv::Point2f(f.opticalFlowNorm.x(), f.opticalFlowNorm.y());
        A.row(i) << p1.x * p2.x, p1.x * p2.y, p1.x, p1.y * p2.x, p1.y * p2.y, p1.y, p2.x, p2.y, 1;
    }

    // The initial solution is found using an SVD of A.
    const Eigen::BDCSVD svdA(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix<ftype, 9, 1> solutionVec = svdA.matrixV().rightCols<1>();
    const Eigen::Matrix3T solutionMat = Eigen::Matrix3T(solutionVec.data()).transpose();

    // The final solution is found by projecting to the nearest essential matrix.
    const Eigen::BDCSVD svdB(solutionMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3T& singularVals = svdB.singularValues();
    const ftype avgSingularVal = 0.5 * (singularVals(0) + singularVals(1));
    const Eigen::Matrix3d essentialMat = svdB.matrixU() *
                                         Eigen::DiagonalMatrix<ftype, 3>(avgSingularVal, avgSingularVal, 0.0) *
                                         svdB.matrixV().transpose();

    return essentialMat;
}

/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <numeric>

#include "eqvio/VIOFilterSettings.h"
#include "eqvio/mathematical/VIO_eqf.h"
#include "testing_utilities.h"
#include "gtest/gtest.h"

constexpr static int numParticles = 1000;

class FilterStatisticsTest : public ::testing::Test {
  protected:
    FilterStatisticsTest() {
        settings.coordinateChoice = CoordinateChoice::InvDepth;
        settings.initialPointVariance = pow(0.01, 2);
        settings.initialPointDepthVariance = pow(0.01, 2);
        settings.initialBiasOmegaVariance = pow(0.01, 2);
        settings.initialBiasAccelVariance = pow(0.01, 2);
        // settings.initialBiasOmegaVariance = pow(0.00001, 2);
        // settings.initialBiasAccelVariance = pow(0.00001, 2);
        // settings.initialAttitudeVariance = pow(0.00001, 2);
        settings.initialVelocityVariance = pow(0.1, 2);
        settings.initialPositionVariance = pow(0.001, 2);
        filter = VIO_eqf{getCoordinates(settings.coordinateChoice), xi0, XHat0, Sigma0()};
        particles = generateInitialStateParticles(Sigma0());

        const cv::Size imageSize = cv::Size(752, 480);
        const double fx = 458.654;
        const double fy = 457.296;
        const double cx = 367.215;
        const double cy = 248.375;
        const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

        camPtr = std::make_shared<GIFT::PinholeCamera>(GIFT::PinholeCamera(imageSize, K));
    }

    // Set up the settings and the initialise the filter.
    const std::vector<int> ids = {0, 1};
    const VIOState xi0 = reasonableStateElement(ids);
    const VIOGroup XHat0 = VIOGroup::Identity(ids);
    VIOFilter::Settings settings;
    VIO_eqf filter;
    std::vector<VIOState> particles;
    GIFT::GICameraPtr camPtr;

    const Eigen::MatrixXd Sigma0() const { return settings.constructInitialStateCovariance(ids.size()); }
    const Eigen::MatrixXd estimateParticleCovariance(const std::vector<VIOState>& particles) const {
        // Assume the mean is already zero.
        Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(xi0.Dim(), xi0.Dim());
        const auto XInv = filter.X.inverse();
        for (const auto& sample : particles) {
            const auto e = stateGroupAction(XInv, sample);
            const auto epsilon = getCoordinates(settings.coordinateChoice)->stateChart(e, xi0);
            const auto sampleCov = epsilon * epsilon.transpose();
            Sigma += sampleCov / particles.size();
        }

        return Sigma;
    }

    const double computeMeanNEES(std::vector<VIOState>& particles) const {
        std::vector<double> NEESValues(numParticles);
        std::transform(particles.begin(), particles.end(), NEESValues.begin(), [this](const VIOState& xi) {
            return filter.computeNEES(xi);
        });

        const double meanNEES = std::accumulate(NEESValues.begin(), NEESValues.end(), 0.0) / numParticles;
        return meanNEES;
    }

    std::vector<VIOState> generateInitialStateParticles(const Eigen::MatrixXd& Sigma) {
        const auto epsilonSamples = sampleGaussianDistribution(Sigma0(), numParticles);
        std::vector<VIOState> particles(numParticles);
        std::transform(epsilonSamples.begin(), epsilonSamples.end(), particles.begin(), [this](const auto& eps) {
            const VIOAlgebra Delta = getCoordinates(settings.coordinateChoice)->liftInnovation(eps, xi0);
            return stateGroupAction(VIOExp(Delta), xi0);
        });
        return particles;
    }
};

TEST_F(FilterStatisticsTest, initialDistribution) { EXPECT_NEAR(computeMeanNEES(particles), 1.0, 0.1); }

TEST_F(FilterStatisticsTest, trueInputDistribution) {
    const double dt = 0.2;
    const IMUVelocity trueVel = IMUVelocity::Zero();
    // const IMUVelocity trueVel = IMUVelocity(Eigen::Matrix<double, 6, 1>::Random().eval());

    for (int rep = 0; rep < 5; ++rep) {
        for (VIOState& xi : particles) {
            xi = integrateSystemFunction(xi, trueVel, dt);
        }

        filter.integrateRiccatiStateDiscrete(
            trueVel, dt, 0 * settings.constructInputGainMatrix(), 0 * settings.constructStateGainMatrix(ids.size()));
        filter.integrateObserverState(trueVel, dt, true);

        EXPECT_NEAR(computeMeanNEES(particles), 1.0, 1.0);
    }
}

TEST_F(FilterStatisticsTest, inputDistribution) {
    // Create a particle distribution
    const double dt = 0.05;
    const IMUVelocity trueVel = IMUVelocity(Eigen::Matrix<double, 6, 1>::Random().eval());
    const auto inputGain = settings.constructInputGainMatrix();

    for (int rep = 0; rep < 5; ++rep) {
        for (VIOState& xi : particles) {
            xi = integrateSystemFunction(xi, trueVel, dt);
        }

        const Eigen::Vector<double, 12> mu = sampleGaussianDistribution(inputGain / dt);
        const IMUVelocity noisyInput = trueVel + mu;
        // Integrate the filter
        filter.integrateRiccatiStateDiscrete(
            trueVel, dt, 0 * settings.constructInputGainMatrix(), 0 * settings.constructStateGainMatrix(ids.size()));
        filter.integrateObserverState(trueVel, dt, true);

        EXPECT_NEAR(computeMeanNEES(particles), 1.0, 0.1);
    }
}

TEST_F(FilterStatisticsTest, outputDistribution) {
    // Compute the probability of each particle given a measurement and resample
    const Eigen::MatrixXd outputGain = settings.constructOutputGainMatrix(ids.size());
    const auto outputGainInv = outputGain.inverse().eval();
    const Eigen::VectorXd outputNoiseSample = sampleGaussianDistribution(outputGain);
    const auto measOutput = measureSystemState(xi0, camPtr) + outputNoiseSample;
    std::vector<double> sampleWeights(numParticles);
    std::transform(
        particles.begin(), particles.end(), sampleWeights.begin(),
        [this, &outputGainInv, &measOutput, &outputNoiseSample](const auto& xi) {
            // Compute the (non-normalised) probability of xi given y
            const auto estOutput = measureSystemState(xi, camPtr);
            Eigen::VectorXd outputError = measOutput - estOutput;
            const double logLikelihood = outputError.transpose() * outputGainInv * outputError;
            const double scaledProbability = std::exp(-0.5 * logLikelihood);
            return scaledProbability;
        });
    const double weightSum = std::accumulate(sampleWeights.begin(), sampleWeights.end(), 0.0);
    for (double& w : sampleWeights) {
        w = w / weightSum;
    }
    const double newWeightSum = std::accumulate(sampleWeights.begin(), sampleWeights.end(), 0.0);
    particles = weightedResample(particles, sampleWeights);

    // Incorporate the measurement into the filter
    filter.performVisionUpdate(measOutput, outputGain);

    EXPECT_NEAR(computeMeanNEES(particles), 1.0, 0.5);
}

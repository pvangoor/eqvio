#include <numeric>

#include "eigen3/Eigen/QR"
#include "opencv2/imgproc.hpp"

#include "eqvio/EqFMatrices.h"
#include "eqvio/LoopTimer.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"

using namespace Eigen;
using namespace std;
using namespace liepp;

Matrix<double, VIOSensorState::CompDim, VIOSensorState::CompDim>
constructBaseSigma(const VIOFilter::Settings& settings) {
    Matrix<double, VIOSensorState::CompDim, VIOSensorState::CompDim> Sigma;
    Sigma.setZero();
    Sigma.block<3, 3>(0, 0) = Matrix3d::Identity() * settings.initialBiasOmegaVariance;
    Sigma.block<3, 3>(3, 3) = Matrix3d::Identity() * settings.initialBiasAccelVariance;
    Sigma.block<3, 3>(6, 6) = Matrix3d::Identity() * settings.initialAttitudeVariance;
    Sigma.block<3, 3>(9, 9) = Matrix3d::Identity() * settings.initialPositionVariance;
    Sigma.block<3, 3>(12, 12) = Matrix3d::Identity() * settings.initialVelocityVariance;
    Sigma.block<3, 3>(15, 15) = Matrix3d::Identity() * settings.initialCameraAttitudeVariance;
    Sigma.block<3, 3>(18, 18) = Matrix3d::Identity() * settings.initialCameraPositionVariance;
    return Sigma;
}

void removeRows(MatrixXd& mat, int startRow, int numRows) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startRow + numRows <= rows);
    mat.block(startRow, 0, rows - numRows - startRow, cols) =
        mat.block(startRow + numRows, 0, rows - numRows - startRow, cols);
    mat.conservativeResize(rows - numRows, NoChange);
}

void removeCols(MatrixXd& mat, int startCol, int numCols) {
    int rows = mat.rows();
    int cols = mat.cols();

    assert(startCol + numCols <= cols);
    mat.block(0, startCol, rows, cols - numCols - startCol) =
        mat.block(0, startCol + numCols, rows, cols - numCols - startCol);
    mat.conservativeResize(NoChange, cols - numCols);
}

VIOFilter::VIOFilter(const VIOFilter::Settings& settings) {
    this->settings = make_unique<VIOFilter::Settings>(settings);
    Sigma = constructBaseSigma(settings);

    xi0.sensor.inputBias.setZero();
    xi0.sensor.pose.setIdentity();
    xi0.sensor.velocity.setZero();
    xi0.sensor.cameraOffset = settings.cameraOffset;

    if (settings.coordinateChoice == CoordinateChoice::Euclidean) {
        coordinateSuite = &EqFCoordinateSuite_euclid;
    } else if (settings.coordinateChoice == CoordinateChoice::InvDepth) {
        coordinateSuite = &EqFCoordinateSuite_invdepth;
    } else if (settings.coordinateChoice == CoordinateChoice::Normal) {
        coordinateSuite = &EqFCoordinateSuite_normal;
    }
}

void VIOFilter::processIMUData(const IMUVelocity& imuVelocity) {
    if (!initialisedFlag) {
        initialiseFromIMUData(imuVelocity);
    }

    integrateUpToTime(imuVelocity.stamp, !settings->fastRiccati);

    // Update the velocity and time
    currentVelocity = imuVelocity;
    currentTime = imuVelocity.stamp;
}

void VIOFilter::initialiseFromIMUData(const IMUVelocity& imuVelocity) {
    xi0.sensor.inputBias.setZero();
    xi0.sensor.pose.setIdentity();
    xi0.sensor.velocity.setZero();
    initialisedFlag = true;

    // Compute the attitude from the gravity vector
    // acc \approx g R^\top e_3,
    // e_3 \approx R acc.normalized()

    const Vector3d& approxGravity = imuVelocity.acc.normalized();
    xi0.sensor.pose.R = SO3d::SO3FromVectors(approxGravity, Vector3d::Unit(2));
}

void VIOFilter::setState(const VIOState& xi) {
    xi0 = xi;
    X = VIOGroup::Identity(xi.getIds());

    const int N = xi.cameraLandmarks.size();
    Sigma = MatrixXd::Identity(VIOSensorState::CompDim + 3 * N, VIOSensorState::CompDim + 3 * N);
    Sigma.block<VIOSensorState::CompDim, VIOSensorState::CompDim>(0, 0) = constructBaseSigma(*settings);
    Sigma.block(VIOSensorState::CompDim, VIOSensorState::CompDim, 3 * N, 3 * N) *= settings->initialPointVariance;

    initialisedFlag = true;
}

bool VIOFilter::integrateUpToTime(const double& newTime, const bool doRiccati) {
    if (currentTime < 0)
        return false;

    const double dt = newTime - currentTime;
    if (dt <= 0)
        return false;

    accumulatedTime += dt;
    accumulatedVelocity = accumulatedVelocity + currentVelocity * dt;

    const int N = xi0.cameraLandmarks.size();
    const VIOState currentState = stateEstimate();

    if (doRiccati && accumulatedTime > 0) {
        assert(!X.hasNaN());
        // Lift the velocity and compute the Riccati process matrices
        MatrixXd PMat = MatrixXd::Identity(Sigma.rows(), Sigma.cols());
        PMat.block<3, 3>(0, 0) *= settings->biasOmegaProcessVariance;
        PMat.block<3, 3>(3, 3) *= settings->biasAccelProcessVariance;
        PMat.block<3, 3>(6, 6) *= settings->attitudeProcessVariance;
        PMat.block<3, 3>(9, 9) *= settings->positionProcessVariance;
        PMat.block<3, 3>(12, 12) *= settings->velocityProcessVariance;
        PMat.block<3, 3>(15, 15) *= settings->cameraAttitudeProcessVariance;
        PMat.block<3, 3>(18, 18) *= settings->cameraPositionProcessVariance;
        PMat.block(VIOSensorState::CompDim, VIOSensorState::CompDim, 3 * N, 3 * N) *= settings->pointProcessVariance;

        accumulatedVelocity = accumulatedVelocity * (1.0 / accumulatedTime);
        const MatrixXd A0t = coordinateSuite->stateMatrixA(X, xi0, accumulatedVelocity);

        // Compute the Riccati velocity matrix
        const MatrixXd Bt = coordinateSuite->inputMatrixB(X, xi0);
        Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim> R =
            Matrix<double, IMUVelocity::CompDim, IMUVelocity::CompDim>::Identity();
        R.block<3, 3>(0, 0) *= settings->velGyrNoise * settings->velGyrNoise;
        R.block<3, 3>(3, 3) *= settings->velAccNoise * settings->velAccNoise;
        R.block<3, 3>(6, 6) *= settings->velGyrBiasWalk * settings->velGyrBiasWalk;
        R.block<3, 3>(9, 9) *= settings->velAccBiasWalk * settings->velAccBiasWalk;

        const auto A0tExp = MatrixXd::Identity(A0t.rows(), A0t.cols()) + accumulatedTime * A0t;
        // Sigma += dt * (PMat + Bt * R * Bt.transpose() + A0t * Sigma + Sigma * A0t.transpose());
        Sigma = accumulatedTime * (PMat + Bt * R * Bt.transpose()) + A0tExp * Sigma * A0tExp.transpose();
        assert(!Sigma.hasNaN());

        accumulatedVelocity = IMUVelocity::Zero();
        accumulatedTime = 0.0;
    }

    // Integrate the equations
    VIOGroup liftedVelocity;
    if (settings->useDiscreteVelocityLift) {
        liftedVelocity = liftVelocityDiscrete(currentState, currentVelocity, dt);
    } else {
        const auto liftedVelocityAlg = liftVelocity(currentState, currentVelocity);
        liftedVelocity = VIOExp(dt * liftedVelocityAlg);
    }
    assert(!liftedVelocity.hasNaN());
    X = X * liftedVelocity;
    assert(!X.hasNaN());

    // removeInvalidLandmarks();

    currentTime = newTime;
    return true;
}

void VIOFilter::processVisionData(const VisionMeasurement& measurement) {
    // Use the stored velocity input to bring the filter up to the current timestamp
    loopTimer.startTiming("propagation");
    bool integrationFlag = integrateUpToTime(measurement.stamp, true);
    if (!integrationFlag || !initialisedFlag)
        return;
    loopTimer.endTiming("propagation");

    loopTimer.startTiming("preprocessing");
    removeOldLandmarks(measurement.getIds());
    assert(measurement.camCoordinates.size() >= X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(measurement.camCoordinates.count(X.id[i]) > 0);
    }

    VisionMeasurement matchedMeasurement = measurement;
    removeOutliers(matchedMeasurement);
    addNewLandmarks(matchedMeasurement);

    assert(matchedMeasurement.camCoordinates.size() == X.id.size());
    for (int i = X.id.size() - 1; i >= 0; --i) {
        assert(matchedMeasurement.camCoordinates.count(X.id[i]) > 0);
    }
    loopTimer.endTiming("preprocessing");

    if (matchedMeasurement.camCoordinates.empty())
        return;

    // --------------------------
    // Compute the EqF innovation
    // --------------------------
    loopTimer.startTiming("correction");
    const VisionMeasurement estimatedMeasurement = measureSystemState(stateEstimate(), measurement.cameraPtr);
    const VisionMeasurement measurementResidual = matchedMeasurement - estimatedMeasurement;
    const MatrixXd Ct = coordinateSuite->outputMatrixC(xi0, X, matchedMeasurement, settings->useEquivariantOutput);
    const int N = xi0.cameraLandmarks.size();
    const MatrixXd QMat = MatrixXd::Identity(2 * N, 2 * N) * settings->measurementNoise * settings->measurementNoise;

    // Use the discrete update form
    const auto& SInv = (Ct * Sigma * Ct.transpose() + QMat).inverse();
    const auto& K = Sigma * Ct.transpose() * SInv;

    const VectorXd yTilde = measurementResidual;
    const VectorXd Gamma = K * yTilde;
    assert(!Gamma.hasNaN());

    VIOGroup Delta;
    if (settings->useDiscreteInnovationLift) {
        Delta = coordinateSuite->liftInnovationDiscrete(Gamma, xi0);
    } else {
        Delta = VIOExp(coordinateSuite->liftInnovation(Gamma, xi0));
    }
    assert(!Delta.hasNaN());

    X = Delta * X;
    Sigma = Sigma - K * Ct * Sigma;

    removeInvalidLandmarks();
    loopTimer.endTiming("correction");

    assert(!Sigma.hasNaN());
    assert(!X.hasNaN());
    // assert(Sigma.eigenvalues().real().minCoeff() > 0);
}

void VIOFilter::removeInvalidLandmarks() {
    set<int> invalidLandmarkIds;
    for (size_t i = 0; i < X.id.size(); ++i) {
        if (X.Q[i].a <= 1e-8 || X.Q[i].a > 1e8) {
            invalidLandmarkIds.emplace(X.id[i]);
        }
    }
    for (const int& lmId : invalidLandmarkIds) {
        removeLandmarkById(lmId);
    }
}

VIOState VIOFilter::stateEstimate() const { return stateGroupAction(this->X, this->xi0); }

VisionMeasurement VIOFilter::getFeaturePredictions(const GIFT::GICameraPtr& camPtr, const double& stamp) {
    if (settings->useFeaturePredictions) {
        if (stamp > 0) {
            integrateUpToTime(stamp, !settings->fastRiccati);
        }
        return measureSystemState(stateEstimate(), camPtr);
    }
    return VisionMeasurement();
}

CSVLine& operator<<(CSVLine& line, const VIOFilter& filter) { return line << filter.xi0 << filter.X << filter.Sigma; }

double VIOFilter::getTime() const { return currentTime; }

void VIOFilter::addNewLandmarks(std::vector<Landmark>& newLandmarks) {
    // Initialise all landmarks to the median scene depth
    double initialDepth = settings->useMedianDepth ? getMedianSceneDepth() : settings->initialSceneDepth;
    for_each(newLandmarks.begin(), newLandmarks.end(), [&initialDepth](Landmark& blm) { blm.p *= initialDepth; });
    xi0.cameraLandmarks.insert(xi0.cameraLandmarks.end(), newLandmarks.begin(), newLandmarks.end());

    vector<int> newIds(newLandmarks.size());
    transform(newLandmarks.begin(), newLandmarks.end(), newIds.begin(), [](const Landmark& blm) { return blm.id; });
    X.id.insert(X.id.end(), newIds.begin(), newIds.end());

    vector<SOT3d> newTransforms(newLandmarks.size());
    for (SOT3d& newTf : newTransforms) {
        newTf.setIdentity();
    }
    X.Q.insert(X.Q.end(), newTransforms.begin(), newTransforms.end());

    const int newN = newLandmarks.size();
    const int ogSize = Sigma.rows();
    Sigma.conservativeResize(ogSize + 3 * newN, ogSize + 3 * newN);
    Sigma.block(ogSize, 0, 3 * newN, ogSize).setZero();
    Sigma.block(0, ogSize, ogSize, 3 * newN).setZero();
    Sigma.block(ogSize, ogSize, 3 * newN, 3 * newN) =
        MatrixXd::Identity(3 * newN, 3 * newN) * settings->initialPointVariance;
}

void VIOFilter::addNewLandmarks(const VisionMeasurement& measurement) {
    // Grab all the new landmarks
    std::vector<Landmark> newLandmarks;
    for (const pair<int, Vector2d>& cc : measurement.camCoordinates) {
        const int& ccId = cc.first;
        if (none_of(X.id.begin(), X.id.end(), [&ccId](const int& i) { return i == ccId; })) {
            Vector3d bearing = measurement.cameraPtr->undistortPoint(cc.second);
            newLandmarks.emplace_back(Landmark{bearing, ccId});
        }
    }
    if (newLandmarks.empty())
        return;
    addNewLandmarks(newLandmarks);
}

void VIOFilter::removeOldLandmarks(const vector<int>& measurementIds) {
    // Determine which indices have been lost
    vector<int> lostIndices(X.id.size());
    iota(lostIndices.begin(), lostIndices.end(), 0);
    if (lostIndices.empty())
        return;

    const auto lostIndicesEnd = remove_if(lostIndices.begin(), lostIndices.end(), [&](const int& lidx) {
        const int& oldId = X.id[lidx];
        return any_of(
            measurementIds.begin(), measurementIds.end(), [&oldId](const int& measId) { return measId == oldId; });
    });
    lostIndices.erase(lostIndicesEnd, lostIndices.end());

    if (lostIndices.empty())
        return;

    // Remove the origin state and transforms and Sigma bits corresponding to these indices.
    reverse(lostIndices.begin(), lostIndices.end()); // Should be in descending order now
    for (const int li : lostIndices) {
        removeLandmarkByIndex(li);
    }
}

void VIOFilter::removeLandmarkByIndex(const int& idx) {
    xi0.cameraLandmarks.erase(xi0.cameraLandmarks.begin() + idx);
    X.id.erase(X.id.begin() + idx);
    X.Q.erase(X.Q.begin() + idx);
    removeRows(Sigma, VIOSensorState::CompDim + 3 * idx, 3);
    removeCols(Sigma, VIOSensorState::CompDim + 3 * idx, 3);
}

void VIOFilter::removeLandmarkById(const int& id) {
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const int idx = distance(xi0.cameraLandmarks.begin(), it);
    removeLandmarkByIndex(idx);
}

void VIOFilter::removeOutliers(VisionMeasurement& measurement) {
    const size_t maxOutliers = (1.0 - settings->featureRetention) * measurement.camCoordinates.size();
    const VIOState xiHat = stateEstimate();
    const VisionMeasurement yHat = measureSystemState(xiHat, measurement.cameraPtr);
    // Remove if the difference between the true and expected measurement exceeds a threshold
    assert(measurement.camCoordinates.size() >= yHat.camCoordinates.size());
    std::vector<int> proposedOutliers;
    std::map<int, double> absoluteOutliers;
    for (const auto& [lmId, yHat_i] : yHat.camCoordinates) {
        assert(measurement.camCoordinates.count(lmId) > 0);
        double bearingErrorAbs = (measurement.camCoordinates[lmId] - yHat_i).norm();
        if (bearingErrorAbs > settings->outlierThresholdAbs) {
            absoluteOutliers[lmId] = bearingErrorAbs;
            proposedOutliers.emplace_back(lmId);
        }
    }

    std::map<int, double> probabilisticOutliers;
    const VisionMeasurement& measurementResidual = measurement - yHat;
    for (const auto& [lmId, yTilde_i] : measurementResidual.camCoordinates) {
        if (absoluteOutliers.count(lmId)) {
            continue;
        }
        const Matrix2d outputCov = getOutputCovById(lmId, measurement.camCoordinates[lmId], measurement.cameraPtr);
        double bearingErrorProb = yTilde_i.transpose() * outputCov.inverse() * yTilde_i;
        if (bearingErrorProb > settings->outlierThresholdProb) {
            probabilisticOutliers[lmId] = bearingErrorProb;
            proposedOutliers.emplace_back(lmId);
        }
    }

    // Prioritise which landmarks to discard
    std::sort(
        proposedOutliers.begin(), proposedOutliers.end(),
        [&absoluteOutliers, &probabilisticOutliers](const int& lmId1, const int& lmId2) {
            if (absoluteOutliers.count(lmId1)) {
                if (absoluteOutliers.count(lmId2)) {
                    return absoluteOutliers.at(lmId1) < absoluteOutliers.at(lmId2);
                } else {
                    return false;
                }
            } else {
                if (absoluteOutliers.count(lmId2)) {
                    return true;
                } else {
                    return probabilisticOutliers.at(lmId1) < probabilisticOutliers.at(lmId2);
                }
            }
        });
    std::reverse(proposedOutliers.begin(), proposedOutliers.end());
    if (proposedOutliers.size() > maxOutliers) {
        proposedOutliers.erase(proposedOutliers.begin() + maxOutliers, proposedOutliers.end());
    }

    for_each(proposedOutliers.begin(), proposedOutliers.end(), [this, &measurement](const int& lmId) {
        removeLandmarkById(lmId);
        measurement.camCoordinates.erase(lmId);
    });
}

double VIOFilter::getMedianSceneDepth() const {
    const vector<Landmark> landmarks = this->stateEstimate().cameraLandmarks;
    vector<double> depthsSquared(landmarks.size());
    transform(landmarks.begin(), landmarks.end(), depthsSquared.begin(), [](const Landmark& blm) {
        return blm.p.squaredNorm();
    });
    const auto midway = depthsSquared.begin() + depthsSquared.size() / 2;
    nth_element(depthsSquared.begin(), midway, depthsSquared.end());
    double medianDepth = settings->initialSceneDepth;
    if (!(midway == depthsSquared.end())) {
        medianDepth = pow(*midway, 0.5);
    }

    return medianDepth;
}

Eigen::Matrix3d VIOFilter::getLandmarkCovById(const int& id) const {
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const int i = distance(xi0.cameraLandmarks.begin(), it);
    return Sigma.block<3, 3>(VIOSensorState::CompDim + 3 * i, VIOSensorState::CompDim + 3 * i);
}

Eigen::Matrix2d VIOFilter::getOutputCovById(const int& id, const Vector2d& y, const GIFT::GICameraPtr& camPtr) const {
    const Matrix3d lmCov = getLandmarkCovById(id);
    const auto it = find_if(
        xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
    assert(it != xi0.cameraLandmarks.end());
    const auto it_X = find_if(X.id.begin(), X.id.end(), [&it](const int& i) { return i == it->id; });
    assert(it_X != X.id.end());
    const SOT3d& Q_i = X.Q[distance(X.id.begin(), it_X)];
    // const Matrix<double, 2, 3> C0i = settings->useEquivariantOutput
    //                                      ? coordinateSuite->outputMatrixCiStar(it->p, Q_i, camPtr, y)
    //                                      : coordinateSuite->outputMatrixCi(it->p, Q_i, camPtr);
    const Matrix<double, 2, 3> C0i = coordinateSuite->outputMatrixCi(it->p, Q_i, camPtr);
    const Matrix2d landmarkCov = C0i * lmCov * C0i.transpose();
    return landmarkCov;
}
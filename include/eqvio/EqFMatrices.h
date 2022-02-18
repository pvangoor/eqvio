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

#pragma once

#include "eqvio/VIOGroup.h"
#include <functional>

/** @file */

/** @brief The suite of functions associated with a choice of coordinates for the EqF.
 *
 * Given the symmetry action of the VIO group, the matrices of the EqF are determined by the choice of local coordinates
 * about the fixed origin. This struct provides pointers to each of those functions, and can be instantiated based on a
 * particular choice of coordinates.
 */
struct EqFCoordinateSuite {
    /** @brief The coordinate chart for the VIO State space, \f$ \vartheta : \mathcal{T}^{\text{VI}}_n(3) \to
     * \mathbb{R}^m \f$.
     */
    const CoordinateChart<VIOState>& stateChart;

    /// The EqF state matrix \f$ \mathring{A}_t \f$
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOState&, const IMUVelocity&)> stateMatrixA;
    /// The EqF input matrix \f$ B_t \f$
    const std::function<Eigen::MatrixXd(const VIOGroup&, const VIOState&)> inputMatrixB;
    /// The EqF equivariant output matrix block \f$ C^\star_t \f$
    const std::function<Eigen::Matrix<double, 2, 3>(
        const Eigen::Vector3d& q0, const liepp::SOT3d& QHat, const GIFT::GICameraPtr& camPtr, const Eigen::Vector2d& y)>
        outputMatrixCiStar;

    /// The output matrix  with equivariance \f$ C_t^\star \f$ or without \f$ C_t \f$.
    const Eigen::MatrixXd outputMatrixC(
        const VIOState& xi0, const VIOGroup& X, const VisionMeasurement& y, const bool useEquivariance = true) const;

    /// The standard (not equivariant) output matrix block \f$ C_i \f$.
    const Eigen::Matrix<double, 2, 3>
    outputMatrixCi(const Eigen::Vector3d& q0, const liepp::SOT3d& QHat, const GIFT::GICameraPtr& camPtr) const;

    /// The continuous-time lift of the correction term \f$ \Delta \f$ from the tangent space at \f$ \mathring{\xi} \f$
    /// to the Lie algebra \f$ \mathfrak{g} \f$.
    const std::function<VIOAlgebra(const Eigen::VectorXd&, const VIOState&)> liftInnovation;
    /// The discrete-time lift of the correction term \f$ \Delta \f$ from the tangent space at \f$ \mathring{\xi} \f$ to
    /// the Lie algebra \f$ \mathfrak{g} \f$.
    const std::function<VIOGroup(const Eigen::VectorXd&, const VIOState&)> liftInnovationDiscrete;
};

/// A suite of functions for the EqF using Euclidean coordinates
extern const EqFCoordinateSuite EqFCoordinateSuite_euclid;
/// A suite of functions for the EqF using Inverse Depth coordinates
extern const EqFCoordinateSuite EqFCoordinateSuite_invdepth;
/// A suite of functions for the EqF using Normal coordinates
extern const EqFCoordinateSuite EqFCoordinateSuite_normal;

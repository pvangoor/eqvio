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

/** @file */

#include "eigen3/Eigen/Core"

#include <functional>
#include <type_traits>

#if EIGEN_MAJOR_VERSION == 3 && EIGEN_MINOR_VERSION <= 9
namespace Eigen {
template <typename _Scalar, int _Rows> using Vector = Matrix<_Scalar, _Rows, 1>;
}
#endif

#if EQVIO_SUPPORT_CONCEPTS
#include <concepts>
// Concept of manifold requires that the class has an int member called `CompDim'.
template <typename T> concept Manifold = requires(T t) { std::is_same<decltype(&T::CompDim, char(0)), int>::value; };
#endif

/** @brief A Coordiante chart for a given manifold M.
 *
 * The manifold M must have a static int CompDim that indicates the dimension of the manifold at compile time (-1 for
 * variable dimension). The coordinate chart must be implemented with a forward and an inverse method.
 */
#if EQVIO_SUPPORT_CONCEPTS
template <Manifold M> struct CoordinateChart {
#else
template <typename M> struct CoordinateChart {
#endif
    /// Function pointer to the implementation of the forward chart
    const std::function<Eigen::Matrix<double, M::CompDim, 1>(const M&, const M&)> chart;
    /// Function pointer to the implementation of the inverse chart
    const std::function<M(const Eigen::Vector<double, M::CompDim>&, const M&)> chartInv;

    /** @brief Apply the forward coordinate chart
     *
     * @param xi The manifold point requiring coordinates.
     * @param xi0 The manifold point that serves as the chart origin.
     * @return the local coordinate of xi about xi0.
     *
     * A coordinate chart maps a point xi near xi0 to an Eigen::Vector. xi0 is the origin of the chart in the sense that
     * chart(xi0, xi0) = 0.
     */
    Eigen::Vector<double, M::CompDim> operator()(const M& xi, const M& xi0) const { return chart(xi, xi0); };

    /** @brief Apply the inverse coordinate chart
     *
     * @param x The Vector to be mapped to the manifold.
     * @param xi0 The manifold point that serves as the chart origin.
     * @return the manifold point xi such that chart(xi, xi0) = x.
     *
     * The inverse coordinate chart maps a real vector x to a point xi near xi0 on the manifold. xi0 is the origin of
     * the chart in the sense that chart^{-1}(0, xi0) = xi0.
     */
    M inv(const Eigen::Vector<double, M::CompDim>& x, const M& xi0) const { return chartInv(x, xi0); };
};

/** @brief A Coordinate chart for an embedded manifold.
 *
 * Given a manifold embedded in R^E, this struct is used to implement coordinate charts R^E -> R^m, where m is the
 * dimension of the manifold itself.
 */
template <int EDim, int Dim = Eigen::Dynamic> struct EmbeddedCoordinateChart {
    using EManifold = Eigen::Matrix<double, EDim, 1>; ///< The embedded manifold type
    /// Function pointer to the implementation of the forward chart
    const std::function<Eigen::Matrix<double, Dim, 1>(const EManifold&, const EManifold&)> chart;
    /// Function pointer to the implementation of the inverse chart
    const std::function<EManifold(const Eigen::Vector<double, Dim>&, const EManifold&)> chartInv;
    /// Function pointer to the implementation of the forward chart differential at 0
    const std::function<Eigen::Matrix<double, Dim, EDim>(const EManifold&)> chartDiff0;
    /// Function pointer to the implementation of the inverse chart differential at 0
    const std::function<Eigen::Matrix<double, EDim, Dim>(const EManifold&)> chartInvDiff0;

    /** @brief Apply the forward coordinate chart
     *
     * @param xi The embedded manifold point requiring coordinates.
     * @param xi0 The embedded manifold point that serves as the chart origin.
     * @return the local coordinate of xi about xi0.
     *
     * A coordinate chart maps a point xi near xi0 to an Eigen::Vector. xi0 is the origin of the chart in the sense that
     * chart(xi0, xi0) = 0.
     */
    Eigen::Vector<double, Dim> operator()(const EManifold& xi, const EManifold& xi0) const { return chart(xi, xi0); };

    /** @brief Apply the inverse coordinate chart
     *
     * @param x The Vector to be mapped to the manifold.
     * @param xi0 The embedded manifold point that serves as the chart origin.
     * @return the embedded manifold point xi such that chart(xi, xi0) = x.
     *
     * The inverse coordinate chart maps a real vector x to a point xi near xi0 on the manifold. xi0 is the origin of
     * the chart in the sense that chart^{-1}(0, xi0) = xi0.
     */
    EManifold inv(const Eigen::Vector<double, Dim>& x, const EManifold& xi0) const { return chartInv(x, xi0); };

    /** @brief Obtain the differential of the forward chart at a given origin.
     *
     * @param xi0 The origin at which the differential is to be computed.
     * @return the differential of the forward chart about xi0.
     */
    Eigen::Matrix<double, Dim, EDim> diff0(const EManifold& xi0) const { return chartDiff0(xi0); };
    /** @brief Obtain the differential of the inverse chart at a given origin.
     *
     * @param xi0 The origin at which the differential is to be computed.
     * @return the differential of the inverse chart about xi0.
     */
    Eigen::Matrix<double, EDim, Dim> invDiff0(const EManifold& xi0) const { return chartInvDiff0(xi0); };
};

/** @brief Compute the numerical differential of a multivariate function.
 *
 * @param f The function R^n -> R^m to be differentiated.
 * @param x The point in R^n where the derivative is to be taken.
 * @param h The step size used to calculate the differential. Optional.
 * @return The differential Df(x) as an m by n matrix.
 *
 * This function computes the numerical differential of a function f at a point x by using the central difference
 * approximation. By default, the step size is computed as the cube root of std::numeric_limits<double>::epsilon(). The
 * optimal step size will depend on the exact function being differentiated, but this is a good starting point most of
 * the time.
 */
Eigen::MatrixXd numericalDifferential(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f, const Eigen::VectorXd& x, double h = -1.0);
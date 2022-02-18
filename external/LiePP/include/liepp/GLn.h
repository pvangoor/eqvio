/*
    This file is part of LiePP.

    LiePP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LiePP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LiePP.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "LieGroup.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"

namespace liepp {

template <int n, typename _Scalar = double> class GLn {
    // The special linear group of n dimensions.
    // n by n matrices with determinant 1.
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = n * n;
    using VectorNS = Eigen::Matrix<_Scalar, n, 1>;
    using MatrixNS = Eigen::Matrix<_Scalar, n, n>;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;

    static MatrixNS wedge(const VectorDS& u) {
        MatrixNS M;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                M(i, j) = u(n * i + j);
            }
        }
        return M;
    }

    static VectorDS vee(const MatrixNS& M) {
        VectorDS u;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                u(n * i + j) = M(i, j);
            }
        }
        return u;
    }

    static MatrixDS adjoint(const VectorDS& u) {
        const auto uWedge = wedge(u);
        MatrixDS adMat;
        for (int i = 0; i < CDim; ++i) {
            const auto eiWedge = wedge(VectorDS::Unit(i));
            adMat.template block<CDim, 1>(0, i) = vee(uWedge * eiWedge - eiWedge * uWedge);
        }
        return adMat;
    }

    static GLn exp(const VectorDS& u) { return GLn(wedge(u).exp()); }

    static VectorDS log(const GLn& X) { return vee(X.asMatrix().log()); }

    static GLn Identity() { return GLn(MatrixNS::Identity()); }
    static GLn Random() {
        MatrixNS M;
        _Scalar d;
        do {
            M.setRandom();
            d = M.determinant();
        } while (abs(d) == 0);
        return GLn(M / d);
    }

    GLn() = default;
    GLn(const MatrixNS& mat) { A = mat; }
    GLn inverse() const { return GLn(A.inverse()); }

    MatrixDS Adjoint() const {
        MatrixNS AInv = A.inverse();
        MatrixDS AdMat;
        for (int i = 0; i < CDim; ++i) {
            const auto ei = VectorDS::Unit(i);
            AdMat.template block<CDim, 1>(0, i) = vee(A * wedge(ei) * AInv);
        }
        return AdMat;
    }

    void setIdentity() { A = MatrixNS::Identity(); }
    VectorDS operator*(const VectorNS& point) const { return A * point; }
    GLn operator*(const GLn& other) const { return GLn(A * other.A); }
    VectorDS applyInverse(const VectorDS& point) const { return A.inverse() * point; }

    void invert() { A = A.inverse(); }

    // Set and get
    MatrixNS asMatrix() const { return A; }
    void fromMatrix(const MatrixNS& mat) { A = mat; }

    static_assert(isLieGroup<GLn<n, _Scalar>>);
    MatrixNS A;
};

template <int n> using GLnd = GLn<n, double>;
template <int n> using GLnf = GLn<n, float>;
template <int n> using GLncd = GLn<n, Eigen::dcomplex>;
template <int n> using GLncf = GLn<n, Eigen::scomplex>;

} // namespace liepp
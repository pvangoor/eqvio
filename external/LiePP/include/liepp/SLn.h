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

template <int n, typename _Scalar = double> class SLn {
    // The special linear group of n dimensions.
    // n by n matrices with determinant 1.
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = n * n - 1;
    using VectorNS = Eigen::Matrix<_Scalar, n, 1>;
    using MatrixNS = Eigen::Matrix<_Scalar, n, n>;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;

    static MatrixNS wedge(const VectorDS& u) {
        MatrixNS M;
        M(n - 1, n - 1) = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (n * i + j < CDim) {
                    M(i, j) = u(n * i + j);

                    if (i == j) {
                        M(n - 1, n - 1) -= u(n * i + j);
                    }
                }
            }
        }
        assert(abs(M.trace()) < 1e-8);
        return M;
    }

    static VectorDS vee(const MatrixNS& M) {
        VectorDS u;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (n * i + j < CDim)
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

    static SLn exp(const VectorDS& u) { return SLn(wedge(u).exp()); }

    static VectorDS log(const SLn& X) { return vee(X.asMatrix().log()); }

    static SLn Identity() { return SLn(MatrixNS::Identity()); }
    static SLn Random() {
        MatrixNS M;
        _Scalar d;
        do {
            M.setRandom();
            d = M.determinant();
        } while (d == 0);
        return SLn(M / d);
    }

    SLn() = default;
    SLn(const MatrixNS& mat) { H = mat; }
    SLn inverse() const { return SLn(H.inverse()); }

    MatrixDS Adjoint() const {
        MatrixNS HInv = H.inverse();
        MatrixDS AdMat;
        for (int i = 0; i < CDim; ++i) {
            const auto ei = VectorDS::Unit(i);
            AdMat.template block<CDim, 1>(0, i) = vee(H * wedge(ei) * HInv);
        }
        return AdMat;
    }

    void setIdentity() { H = MatrixNS::Identity(); }
    VectorDS operator*(const VectorNS& point) const { return H * point; }
    SLn operator*(const SLn& other) const { return SLn(H * other.H); }
    VectorDS applyInverse(const VectorDS& point) const { return H.inverse() * point; }

    void invert() { H = H.inverse(); }

    // Set and get
    MatrixNS asMatrix() const { return H; }
    void fromMatrix(const MatrixNS& mat) { H = mat; }

    MatrixNS H;
    static_assert(isLieGroup<SLn<n, _Scalar>>);
};

using SL3d = SLn<3, double>;
using SL3f = SLn<3, float>;
using SL3cd = SLn<3, Eigen::dcomplex>;
using SL3cf = SLn<3, Eigen::scomplex>;

}
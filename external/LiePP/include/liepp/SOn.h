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

namespace liepp {

template <int n, typename _Scalar = double> class SOn {
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = (n * (n - 1)) / 2;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;
    using MatrixNS = Eigen::Matrix<_Scalar, n, n>;
    using VectorNS = Eigen::Matrix<_Scalar, n, 1>;

    static MatrixNS wedge(const VectorDS& v) {
        MatrixNS vWedge = MatrixNS::Zero();
        int i = 0;
        int j = 0;
        for (int k = 0; k < CDim; ++k) {
            if (++j >= n) {
                ++i;
                j = i + 1;
            }
            vWedge(i, j) = v(k);
            vWedge(j, i) = -v(k);
        }
        return vWedge;
    }
    static VectorDS vee(const MatrixNS& M) {
        VectorDS v;
        int i = 0;
        int j = 0;
        for (int k = 0; k < CDim; ++k) {
            if (++j >= n) {
                ++i;
                j = i + 1;
            }
            v(k) = M(i, j);
        }
        return v;
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

    static SOn exp(const VectorDS& u) { return SOn(wedge(u).exp()); }

    static VectorDS log(const SOn& X) { return vee(X.asMatrix().log()); }

    static SOn Identity() { return SOn(MatrixNS::Identity()); }
    static SOn Random() { return SOn(MatrixNS::Random()); }

    SOn() = default;
    SOn(const MatrixNS& mat) {
        // Project to nearest orthogonal matrix
        Eigen::BDCSVD<MatrixNS> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
    }
    SOn inverse() const {
        SOn result;
        result.R = R.transpose();
        return result;
    }

    MatrixDS Adjoint() const {
        MatrixNS RT = R.transpose();
        MatrixDS AdMat;
        for (int i = 0; i < CDim; ++i) {
            const auto ei = VectorDS::Unit(i);
            AdMat.template block<CDim, 1>(0, i) = vee(R * wedge(ei) * RT);
        }
        return AdMat;
    }

    void setIdentity() { R = MatrixNS::Identity(); }
    VectorNS operator*(const VectorNS& point) const { return R * point; }
    SOn operator*(const SOn& other) const { return SOn(R * other.R); }
    VectorNS applyInverse(const VectorNS& point) const { return R.transpose() * point; }

    void invert() { R = R.transpose(); }

    // Set and get
    MatrixNS asMatrix() const { return R; }
    void fromMatrix(const MatrixNS& mat) { R = mat; }

    MatrixNS R;
    static_assert(isLieGroup<SOn<n, _Scalar>>);
};

template <int n> using SOnd = SOn<n, double>;
template <int n> using SOnf = SOn<n, float>;
template <int n> using SOncd = SOn<n, Eigen::dcomplex>;
template <int n> using SOncf = SOn<n, Eigen::scomplex>;

}
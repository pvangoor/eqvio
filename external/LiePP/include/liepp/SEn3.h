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

#include "SO3.h"
#include <array>

namespace liepp {

template <int n, typename _Scalar = double> class SEn3 {
  public:
    using Scalar = _Scalar;
    constexpr static int CDim = 3 + 3 * n;
    using Vector3S = Eigen::Matrix<_Scalar, 3, 1>;
    using Matrix3S = Eigen::Matrix<_Scalar, 3, 3>;
    using MatrixNS = Eigen::Matrix<_Scalar, 3 + n, 3 + n>;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;
    using SO3S = SO3<_Scalar>;

    static MatrixNS wedge(const VectorDS& u) {
        // u is in the format (omega, v_1, ..., v_n)
        MatrixNS result;
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template block<3, 1>(0, 0));
        for (int i = 0; i < n; ++i) {
            result.template block<3, 1>(0, 3 + i) = u.template block<3, 1>(3 + 3 * i, 0);
        }
        result.template block<n, 3 + n>(3, 0) = Eigen::Matrix<_Scalar, n, 3 + n>::Zero();
        return result;
    }
    static VectorDS vee(const MatrixNS& U) {
        // u is in the format (omega, v)
        VectorDS result;
        result.template block<3, 1>(0, 0) = SO3S::vex(U.template block<3, 3>(0, 0));
        for (int i = 0; i < n; ++i) {
            result.template block<3, 1>(3 + 3 * i, 0) = U.template block<3, 1>(0, 3 + i);
        }
        return result;
    }
    static MatrixDS adjoint(const VectorDS& u) {
        // u is in the format (omega, v)
        MatrixDS result = MatrixDS::Zero();
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template segment<3>(0));
        for (int i = 0; i < n; ++i) {
            result.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = SO3S::skew(u.template segment<3>(0));
            result.template block<3, 3>(3 + 3 * i, 0) = SO3S::skew(u.template segment<3>(3 + 3 * i));
        }
        return result;
    }
    static SEn3 exp(const VectorDS& u) {
        Vector3S w = u.template block<3, 1>(0, 0);
        std::array<Vector3S, n> v;
        for (int i = 0; i < n; ++i) {
            v[i] = u.template segment<3>(3 + 3 * i);
        }

        _Scalar th = w.norm();
        _Scalar A, B, C;
        if (abs(th) >= 1e-12) {
            A = sin(th) / th;
            B = (1 - cos(th)) / pow(th, 2);
            C = (1 - A) / pow(th, 2);
        } else {
            A = 1.0;
            B = 1.0 / 2.0;
            C = 1.0 / 6.0;
        }

        Matrix3S wx = SO3S::skew(w);
        Matrix3S R = Matrix3S::Identity() + A * wx + B * wx * wx;
        Matrix3S V = Matrix3S::Identity() + B * wx + C * wx * wx;

        SEn3 result;
        result.R = SO3S(R);
        std::transform(v.begin(), v.end(), result.x.begin(), [&V](const Vector3S& v_i) { return V * v_i; });

        return result;
    }
    static VectorDS log(const SEn3& P) {
        Matrix3S Omega = SO3S::skew(SO3S::log(P.R));

        _Scalar theta = SO3S::vex(Omega).norm();
        _Scalar coefficient = 1.0 / 12.0;
        if (abs(theta) > 1e-8) {
            coefficient = 1 / (theta * theta) * (1 - (theta * sin(theta)) / (2 * (1 - cos(theta))));
        }

        Matrix3S VInv = Matrix3S::Identity() - 0.5 * Omega + coefficient * Omega * Omega;

        VectorDS u = VectorDS::Zero();
        u.template segment<3>(0) = SO3S::vex(Omega);
        for (int i = 0; i < n; ++i) {
            u.template segment<3>(3 + 3 * i) = VInv * P.x[i];
        }

        return u;
    }
    static SEn3 Identity() {
        SEn3 result;
        result.R.setIdentity();
        std::for_each(result.x.begin(), result.x.end(), [](Vector3S& x_i) { x_i.setZero(); });
        return result;
    }
    static SEn3 Random() {
        SEn3 result;
        result.R = SO3S::Random();
        std::for_each(result.x.begin(), result.x.end(), [](Vector3S& x_i) { x_i.setRandom(); });
        return result;
    }

    SEn3() = default;
    SEn3(const SEn3& other) {
        R = other.R;
        x = other.x;
    }
    SEn3(const MatrixNS& mat) {
        R = SO3S(mat.template block<3, 3>(0, 0));
        for (int i = 0; i < n; ++i) {
            x[i] = mat.template block<3, 1>(0, 3 + i);
        }
    }
    SEn3(const SO3S& R, const std::array<Vector3S, n>& x) {
        this->R = R;
        this->x = x;
    }

    void setIdentity() {
        R.setIdentity();
        for (int i = 0; i < n; ++i) {
            x[i].setZero();
        }
    }
    std::array<Vector3S, n> operator*(const std::array<Vector3S, n>& points) const {
        std::array<Vector3S, n> result;
        for (int i = 0; i < n; ++i) {
            result[i] = R * points[i] + x[i];
        }
        return result;
    }
    SEn3 operator*(const SEn3& other) const {
        SEn3 result;
        result.R = R * other.R;
        for (int i = 0; i < n; ++i) {
            result.x[i] = R * other.x[i] + x[i];
        }
        return result;
    }

    void invert() {
        for (int i = 0; i < n; ++i) {
            x[i] = -R.inverse() * x[i];
        }
        R = R.inverse();
    }
    SEn3 inverse() const {
        SEn3 result;
        result.R = R.inverse();
        for (int i = 0; i < n; ++i) {
            result.x[i] = -(result.R * x[i]);
        }
        return result;
    }
    MatrixDS Adjoint() const {
        MatrixDS AdMat = MatrixDS::Zero();
        Matrix3S Rmat = R.asMatrix();
        AdMat.template block<3, 3>(0, 0) = Rmat;
        for (int i = 0; i < n; ++i) {
            AdMat.template block<3, 3>(3 + 3 * i, 0) = SO3S::skew(x[i]) * Rmat;
            AdMat.template block<3, 3>(3 + 3 * i, 3 + 3 * i) = Rmat;
        }
        return AdMat;
    }

    // Set and get
    MatrixNS asMatrix() const {
        MatrixNS result;
        result.setIdentity();
        result.template block<3, 3>(0, 0) = R.asMatrix();
        for (int i = 0; i < n; ++i) {
            result.template block<3, 1>(0, 3 + i) = x[i];
        }
        return result;
    }
    void fromMatrix(const MatrixNS& mat) {
        R.fromMatrix(mat.template block<3, 3>(0, 0));
        for (int i = 0; i < n; ++i) {
            x[i] = mat.template block<3, 1>(0, 3 + i);
        }
    }

    SO3S R;
    std::array<Vector3S, n> x;
    static_assert(isLieGroup<SEn3<n, _Scalar>>);
};

template <int n> using SEn3d = SEn3<n, double>;
template <int n> using SEn3f = SEn3<n, float>;
template <int n> using SEn3cd = SEn3<n, Eigen::dcomplex>;
template <int n> using SEn3cf = SEn3<n, Eigen::scomplex>;

using SE23d = SEn3<2, double>;
using SE23f = SEn3<2, float>;
using SE23cd = SEn3<2, Eigen::dcomplex>;
using SE23cf = SEn3<2, Eigen::scomplex>;

}
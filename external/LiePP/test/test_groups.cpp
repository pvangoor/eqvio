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

#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "liepp/GLn.h"
#include "liepp/SE3.h"
#include "liepp/SEn3.h"
#include "liepp/SLn.h"
#include "liepp/SO3.h"
#include "liepp/SOT3.h"
#include "liepp/SOn.h"
#include "gtest/gtest.h"

#include <type_traits>
template <class T> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

using namespace std;
using namespace Eigen;
using namespace liepp;

template <typename _Scalar, std::enable_if_t<is_complex<_Scalar>::value, bool> = true>
void expectNear(const _Scalar& a, const _Scalar& b, const double abs_error = 1e-8) {
    EXPECT_NEAR(a.real(), b.real(), abs_error);
    EXPECT_NEAR(a.imag(), b.imag(), abs_error);
}

template <typename _Scalar, std::enable_if_t<!is_complex<_Scalar>::value, bool> = true>
void expectNear(const _Scalar& a, const _Scalar& b, const double abs_error = 1e-8) {
    EXPECT_NEAR(a, b, abs_error);
}

template <typename _Scalar, int rows, int cols>
void testMatrixEquality(
    const Eigen::Matrix<_Scalar, rows, cols>& M1, const Eigen::Matrix<_Scalar, rows, cols>& M2,
    const double abs_error = 1e-8) {
    for (int i = 0; i < M1.rows(); ++i) {
        for (int j = 0; j < M1.cols(); ++j) {
            expectNear(M1(i, j), M2(i, j), abs_error);
        }
    }
}

TEST(TestGroups, SO3FromVectors) {
    // Test generating an SO(3) matrix between two vectors
    for (int i = 0; i < 100; ++i) {
        Vector3d v = Vector3d::Random();
        Vector3d w = Vector3d::Random();

        v = v.normalized();
        w = w.normalized();

        Matrix3d R = SO3d::SO3FromVectors(v, w).asMatrix();
        Vector3d w2 = R * v;

        testMatrixEquality(w, w2);
    }
}

template <typename T> class MatrixGroupTest : public testing::Test {};

using testing::Types;
typedef Types<SO3d, SE3d, SOT3d, SE23d, SL3d, GLnd<5>, GLncd<3>, SOnd<4>> MatrixGroups;

TYPED_TEST_SUITE(MatrixGroupTest, MatrixGroups);

TYPED_TEST(MatrixGroupTest, TestExpLog) {
    // Test the matrix group exponential and logarithm
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorDS v = TypeParam::VectorDS::Random();

        auto X1 = TypeParam::exp(v).asMatrix();
        decltype(X1) X2 = TypeParam::wedge(v).exp();

        testMatrixEquality(X1, X2);

        auto v11 = TypeParam::log(TypeParam(X1));
        auto v12 = TypeParam::vee(X1.log());
        auto v21 = TypeParam::log(TypeParam(X2));
        auto v22 = TypeParam::vee(X2.log());

        testMatrixEquality(v, v11);
        testMatrixEquality(v, v12);
        testMatrixEquality(v, v21);
        testMatrixEquality(v, v22);
    }
}

TYPED_TEST(MatrixGroupTest, TestWedgeVee) {
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorDS v = TypeParam::VectorDS::Random();
        typename TypeParam::MatrixNS vWedge = TypeParam::wedge(v);
        typename TypeParam::VectorDS vWedgeVee = TypeParam::vee(vWedge);
        testMatrixEquality(vWedgeVee, v);
    }
}

TYPED_TEST(MatrixGroupTest, TestAssociativity) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X1 = TypeParam::Random();
        TypeParam X2 = TypeParam::Random();
        TypeParam X3 = TypeParam::Random();

        TypeParam Z1 = (X1 * X2) * X3;
        TypeParam Z2 = X1 * (X2 * X3);

        testMatrixEquality(Z1.asMatrix(), Z2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestIdentity) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        TypeParam I = TypeParam::Identity();

        TypeParam X1 = X * I;
        TypeParam X2 = I * X;

        testMatrixEquality(X.asMatrix(), X1.asMatrix());
        testMatrixEquality(X.asMatrix(), X2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestInverse) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        TypeParam XInv = X.inverse();
        TypeParam I = TypeParam::Identity();

        TypeParam I1 = X * XInv;
        TypeParam I2 = XInv * X;

        testMatrixEquality(I.asMatrix(), I1.asMatrix());
        testMatrixEquality(I.asMatrix(), I2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixGroupAdjoint) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        typename TypeParam::VectorDS U = TypeParam::VectorDS::Random();

        typename TypeParam::MatrixNS Ad_XU1 = TypeParam::wedge(X.Adjoint() * U);
        typename TypeParam::MatrixNS Ad_XU2 = X.asMatrix() * TypeParam::wedge(U) * X.inverse().asMatrix();

        testMatrixEquality(Ad_XU1, Ad_XU2);
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixAlgebraAdjoint) {
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorDS V = TypeParam::VectorDS::Random();
        typename TypeParam::VectorDS U = TypeParam::VectorDS::Random();

        typename TypeParam::MatrixNS ad_VU1 = TypeParam::wedge(TypeParam::adjoint(V) * U);
        typename TypeParam::MatrixNS ad_VU2 =
            TypeParam::wedge(V) * TypeParam::wedge(U) - TypeParam::wedge(U) * TypeParam::wedge(V);

        testMatrixEquality(ad_VU1, ad_VU2);
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixProduct) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X1 = TypeParam::Random();
        TypeParam X2 = TypeParam::Random();

        typename TypeParam::MatrixNS Z1 = X1.asMatrix() * X2.asMatrix();
        typename TypeParam::MatrixNS Z2 = (X1 * X2).asMatrix();

        testMatrixEquality(Z1, Z2);
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixIdentity) {
    typename TypeParam::MatrixNS I1 = TypeParam::MatrixNS::Identity();
    typename TypeParam::MatrixNS I2 = TypeParam::Identity().asMatrix();

    testMatrixEquality(I1, I2);
}

TYPED_TEST(MatrixGroupTest, TestMatrixInverse) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();

        typename TypeParam::MatrixNS XInv1 = X.inverse().asMatrix();
        typename TypeParam::MatrixNS XInv2 = X.asMatrix().inverse();

        testMatrixEquality(XInv1, XInv2);
    }
}
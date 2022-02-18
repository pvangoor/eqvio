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

#include "GIFT/OptimiseParameters.h"
#include "GIFT/ParameterGroup.h"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace GIFT;

int GIFT::clamp(const int x, const int a, const int b) {
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image);
MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch);
VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image);

void GIFT::optimiseParameters(vector<ParameterGroup>& params, const vector<PyramidPatch>& patches, const Mat& image) {
    if (patches.size() == 0)
        return;
    optimiseParameters(params, patches, ImagePyramid(image, patches[0].levels.size()));
}

void GIFT::optimiseParameters(
    vector<ParameterGroup>& params, const vector<PyramidPatch>& patches, const ImagePyramid& pyramid) {
    assert(patches.size() == params.size());
    for (int i = 0; i < params.size(); ++i) {
        optimiseParameters(params[i], patches[i], pyramid);
    }
}

void GIFT::optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid) {
    const int numLevels = patch.levels.size();
    for (int lv = numLevels - 1; lv >= 0; --lv) {
        params.changeLevel(lv);
        optimiseParametersAtLevel(params, patch.levels[lv], pyramid.levels[lv]);
    }
    params.changeLevel(0);
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    // Use the Inverse compositional algorithm to optimise params at the given level.
    const MatrixXT jacobian = patchActionJacobian(params, patch);
    const MatrixXT stepOperator = (jacobian.transpose() * jacobian).inverse() * jacobian.transpose();
    VectorXT previousStepDirection = VectorXT::Zero(params.dim());

    for (int iteration = 0; iteration < 50; ++iteration) {
        MatrixXT residualVector = paramResidual(params, patch, image);
        MatrixXT stepVector = -stepOperator * residualVector;
        if (stepVector.norm() < 1e-3)
            break;

        VectorXT stepDirection = stepVector.normalized();
        if (stepDirection.dot(previousStepDirection) < -0.5)
            stepVector = stepVector / 2.0;
        previousStepDirection = stepDirection;

        params.applyStepOnRight(stepVector);
    }
}

MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch) {
    // The patch is vectorised row by row.
    const Vector2T offset = 0.5 * Vector2T(patch.cols() - 1, patch.rows() - 1);
    MatrixXT jacobian(patch.area(), params.dim());
    for (int y = 0; y < patch.rows(); ++y) {
        for (int x = 0; x < patch.cols(); ++x) {
            const Vector2T point = Vector2T(x, y) - offset;
            int rowIdx = (x + y * patch.rows());

            jacobian.block(rowIdx, 0, 1, params.dim()) = patch.differential(y, x) * params.actionJacobian(point);
        }
    }
    return jacobian;
}

VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    const Vector2T offset = 0.5 * Vector2T(patch.rows() - 1, patch.cols() - 1);
    VectorXT residualVector = VectorXT(patch.rows() * patch.cols());
    for (int y = 0; y < patch.rows(); ++y) {
        for (int x = 0; x < patch.cols(); ++x) {
            const Vector2T point = Vector2T(x, y) - offset;
            const Vector2T transformedPoint = patch.centre + params.applyLeftAction(point);
            const float subPixelValue = getSubPixel(image, transformedPoint);
            residualVector(x + y * patch.rows()) = subPixelValue - patch.at(y, x);
        }
    }
    return residualVector;
}

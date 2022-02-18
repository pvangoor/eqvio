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

#include "eqvio/Geometry.h"

Eigen::MatrixXd
numericalDifferential(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f, const Eigen::VectorXd& x, double h) {
    if (h < 0) {
        h = std::cbrt(std::numeric_limits<double>::epsilon());
    }
    Eigen::MatrixXd Df(f(x).rows(), x.rows());
    for (int j = 0; j < Df.cols(); ++j) {
        const Eigen::VectorXd ej = Eigen::VectorXd::Unit(Df.cols(), j);
        Df.col(j) = (f(x + h * ej) - f(x - h * ej)) / (2 * h);
    }
    return Df;
}
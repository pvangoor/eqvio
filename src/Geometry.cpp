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
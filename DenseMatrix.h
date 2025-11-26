#pragma once

#include <vector>
#include <string>
#include "COOMatrix.h"

class DenseMatrix {
public:
    DenseMatrix(int rows, int cols, double init = 0.0);
    DenseMatrix(const COOMatrix& coo);

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    double& operator()(int r, int c) { return data_[r * cols_ + c]; }
    double  operator()(int r, int c) const { return data_[r * cols_ + c]; }

    DenseMatrix mul_scalar_p(double alpha) const;
    std::vector<double> mul_vector_p(const std::vector<double>& x) const;
    DenseMatrix transpose_p() const;
    DenseMatrix add_p(const DenseMatrix& other) const;

    DenseMatrix mul_scalar(double alpha) const;
    std::vector<double> mul_vector(const std::vector<double>& x) const;
    DenseMatrix transpose() const;
    DenseMatrix add(const DenseMatrix& other) const;

    void print_matrix() const;
    std::string repr() const;

private:
    int rows_, cols_;
    std::vector<double> data_;
};

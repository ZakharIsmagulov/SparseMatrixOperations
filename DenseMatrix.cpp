#include "DenseMatrix.h"
#include <omp.h>
#include <iostream>
#include <string>
#include "VecRepr.h"

DenseMatrix::DenseMatrix(int rows, int cols, double init)
    : rows_(rows), cols_(cols), data_(rows * cols, init)
{}

DenseMatrix::DenseMatrix(const COOMatrix& coo) {
    rows_ = coo.rows();
    cols_ = coo.cols();
    data_ = std::vector<double>(rows_ * cols_, 0.0);
    const std::vector<int>& r_idx = coo.row_idx();
    const std::vector<int>& c_idx = coo.col_idx();
    const std::vector<double>& v_val = coo.val();
    int nnz = coo.nnz();

    for (int k = 0; k < nnz; ++k) {
        int r = r_idx[k];
        int c = c_idx[k];
        double v = v_val[k];
        data_[r * cols_ + c] = v;
    }
}

DenseMatrix DenseMatrix::mul_scalar_p(double alpha) const {
    DenseMatrix res(rows_, cols_);

    int n = rows_ * cols_;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.data_[i] = data_[i] * alpha;
    }
    return res;
}

std::vector<double> DenseMatrix::mul_vector_p(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        double sum = 0.0;
        int base = r * cols_;
        for (int c = 0; c < cols_; c++) {
            sum += data_[base + c] * x[c];
        }
        res[r] = sum;
    }
    return res;
}

DenseMatrix DenseMatrix::transpose_p() const {
    DenseMatrix res(cols_, rows_);

    // collapse(2), т.к. эти циклы можно развернуть в один большой
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            res(c, r) = data_[r * cols_ + c];
        }
    }
    return res;
}

DenseMatrix DenseMatrix::add_p(const DenseMatrix& other) const {
    DenseMatrix res(rows_, cols_);
    int n = rows_ * cols_;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.data_[i] = data_[i] + other.data_[i];
    }
    return res;
}

DenseMatrix DenseMatrix::mul_scalar(double alpha) const {
    DenseMatrix res(rows_, cols_);

    int n = rows_ * cols_;

    for (int i = 0; i < n; i++) {
        res.data_[i] = data_[i] * alpha;
    }
    return res;
}

std::vector<double> DenseMatrix::mul_vector(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    for (int r = 0; r < rows_; r++) {
        double sum = 0.0;
        int base = r * cols_;
        for (int c = 0; c < cols_; c++) {
            sum += data_[base + c] * x[c];
        }
        res[r] = sum;
    }
    return res;
}

DenseMatrix DenseMatrix::transpose() const {
    DenseMatrix res(cols_, rows_);

    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            res(c, r) = data_[r * cols_ + c];
        }
    }
    return res;
}

DenseMatrix DenseMatrix::add(const DenseMatrix& other) const {
    DenseMatrix res(rows_, cols_);
    int n = rows_ * cols_;

    for (int i = 0; i < n; i++) {
        res.data_[i] = data_[i] + other.data_[i];
    }
    return res;
}

void DenseMatrix::print_matrix() const {
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            std::cout << data_[r * cols_ + c] << " ";
        }
        std::cout << "\n";
    }
}

std::string DenseMatrix::repr() const {
    std::string s;

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            s += format2((*this)(r, c));
            if (c + 1 < cols_)
                s += " ";
        }
        if (r + 1 < rows_)
            s += "\n";
    }

    return s;
}

#pragma once

#include <vector>
#include <string>
#include "COOMatrix.h"

class CSCMatrix {
public:
    CSCMatrix(int rows, int cols);
    CSCMatrix(const COOMatrix& coo);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    // Number of non-zeros
    int nnz()  const { return (int)val_.size(); }

    void add_entry(int r, int c, double v);

    // Закрыть построение
    void end_creation();

    CSCMatrix mul_scalar_p(double alpha) const;
    std::vector<double> mul_vector_p(const std::vector<double>& x) const;
    CSCMatrix transpose_p() const;
    CSCMatrix add_p(const CSCMatrix& other) const;

    CSCMatrix mul_scalar(double alpha) const;
    std::vector<double> mul_vector(const std::vector<double>& x) const;
    CSCMatrix transpose() const;
    CSCMatrix add(const CSCMatrix& other) const;

    void print_matrix() const;
    std::string repr() const;

private:
    int rows_, cols_, last_row_, last_col_;
    std::vector<int> col_ptr_;   // size = cols_ + 1
    std::vector<int> row_idx_;   // size = nnz
    std::vector<double> val_;    // size = nnz
};
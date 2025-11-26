#pragma once

#include <vector>
#include <string>
#include "COOMatrix.h"

class CSRMatrix {
public:
    CSRMatrix(int rows, int cols);
    CSRMatrix(const COOMatrix& coo);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    // Number of non-zeros
    int nnz()  const { return (int)val_.size(); }

    void add_entry(int r, int c, double v);

    // Закрыть построение
    void end_creation();

    CSRMatrix mul_scalar_p(double alpha) const;
    std::vector<double> mul_vector_p(const std::vector<double>& x) const;
    CSRMatrix transpose_p() const;
    CSRMatrix add_p(const CSRMatrix& other) const;

    CSRMatrix mul_scalar(double alpha) const;
    std::vector<double> mul_vector(const std::vector<double>& x) const;
    CSRMatrix transpose() const;
    CSRMatrix add(const CSRMatrix& other) const;

    void print_matrix() const;
    std::string repr() const;

private:
    int rows_, cols_, last_row_, last_col_;
    std::vector<int> row_ptr_;   // size = rows_+1
    std::vector<int> col_idx_;   // size = nnz
    std::vector<double> val_;    // size = nnz
};

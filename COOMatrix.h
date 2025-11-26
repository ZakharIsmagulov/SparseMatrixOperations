#pragma once

#include <vector>
#include <string>

class COOMatrix {
public:
    COOMatrix(int rows, int cols);
    COOMatrix(const std::string& filename);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    // Number of non-zeros
    int nnz()  const { return (int)val_.size(); }
    const std::vector<int>& row_idx() const { return row_idx_; }
    const std::vector<int>& col_idx() const { return col_idx_; }
    const std::vector<double>& val() const { return val_; }

    // Добавить ненулевой элемент
    void add_entry(int r, int c, double v);

    COOMatrix mul_scalar_p(double alpha) const;
    std::vector<double> mul_vector_p(const std::vector<double>& x) const;
    COOMatrix transpose_p() const;
    COOMatrix add_p(const COOMatrix& other) const;

    COOMatrix mul_scalar(double alpha) const;
    std::vector<double> mul_vector(const std::vector<double>& x) const;
    COOMatrix transpose() const;
    COOMatrix add(const COOMatrix& other) const;

    void print_matrix() const;
    std::string repr() const;

private:
    int rows_, cols_;
    std::vector<int> row_idx_;
    std::vector<int> col_idx_;
    std::vector<double> val_;
};

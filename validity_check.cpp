#include <iostream>
#include <vector>
#include "DenseMatrix.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "CSCMatrix.h"
#include "VecRepr.h"

template <typename MatrixType>
void matr_add(const MatrixType& m1, const MatrixType& m11,
              const MatrixType& m2, const MatrixType& m22,
              const MatrixType& m3, const MatrixType& m33,
              const std::string name) {
    std::string m1_ =
        "1.00 0.00 5.00 0.00 2.00 0.00 0.00 3.00 4.00";

    std::string m2_ =
        "0.00 0.00 0.00 1.00 4.00 0.00 0.00\n"
        "0.00 0.00 2.00 0.00 0.00 0.00 0.00\n"
        "7.00 0.00 0.00 0.00 0.00 3.00 0.00\n"
        "0.00 0.00 0.00 2.00 0.00 0.00 4.00\n"
        "5.00 0.00 9.00 0.00 0.00 0.00 1.00";

    std::string m3_ =
        "0.00 1.00 0.00 8.00\n"
        "0.00 3.00 0.00 2.00\n"
        "5.00 0.00 3.00 0.00\n"
        "4.00 0.00 0.00 0.00\n"
        "0.00 7.00 0.00 1.00\n"
        "0.00 5.00 4.00 0.00";

    if (m1_ == m1.add_p(m11).repr() && m2_ == m2.add_p(m22).repr() && m3_ == m3.add_p(m33).repr())
        std::cout << "YES (Matrices for " << name << " adding (parallel) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " adding (parallel) are invalid)\n";

    if (m1_ == m1.add(m11).repr() && m2_ == m2.add(m22).repr() && m3_ == m3.add(m33).repr())
        std::cout << "YES (Matrices for " << name << " adding (sequential) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " adding (sequential) are invalid)\n";
}

template <typename MatrixType>
void matr_transpose(const MatrixType& m1, const MatrixType& m2, const MatrixType& m3, const std::string name) {
    std::string m1_T =
        "0.00\n"
        "0.00\n"
        "5.00\n"
        "0.00\n"
        "0.00\n"
        "0.00\n"
        "0.00\n"
        "3.00\n"
        "0.00";

    std::string m2_T =
        "0.00 0.00 7.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 9.00\n"
        "0.00 0.00 0.00 2.00 0.00\n"
        "4.00 0.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 1.00";

    std::string m3_T =
        "0.00 0.00 5.00 0.00 0.00 0.00\n"
        "0.00 3.00 0.00 0.00 7.00 0.00\n"
        "0.00 0.00 0.00 0.00 0.00 4.00\n"
        "8.00 0.00 0.00 0.00 1.00 0.00";

    if (m1_T == m1.transpose_p().repr() && m2_T == m2.transpose_p().repr() && m3_T == m3.transpose_p().repr())
        std::cout << "YES (Matrices for " << name << " transposing (parallel) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " transposing (parallel) are invalid)\n";

    if (m1_T == m1.transpose().repr() && m2_T == m2.transpose().repr() && m3_T == m3.transpose().repr())
        std::cout << "YES (Matrices for " << name << " transposing (sequential) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " transposing (sequential) are invalid)\n";
}

template <typename MatrixType>
void matr_vec(const MatrixType& m1, const MatrixType& m2, const MatrixType& m3, const std::string name) {
    std::vector<double> m1_y = { -3.4 };

    std::vector<double> m2_y = {
    2.8, 0.0, 0.0, 4.6, -14.3
    };

    std::vector<double> m3_y = {
    29.6, 3.3, -10.0, 0.0, 11.4, 0.0
    };

    std::vector<double> x_m1 = {
    1.0, 0.5, -2.0, 3.1, 0.0, 4.4, -1.2, 2.2, 0.7
    };

    std::vector<double> x_m2 = {
    0.0, 1.2, -1.5, 2.3, 0.7, 3.3, -0.8
    };

    std::vector<double> x_m3 = {
    -2.0, 1.1, 0.0, 3.7
    };

    if (repr_vector(m1_y) == repr_vector(m1.mul_vector_p(x_m1)) 
        && repr_vector(m2_y) == repr_vector(m2.mul_vector_p(x_m2)) 
        && repr_vector(m3_y) == repr_vector(m3.mul_vector_p(x_m3)))
        std::cout << "YES (Matrices for " << name << " multiplying by vector (parallel) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " multiplying by vector (parallel) are invalid)\n";

    if (repr_vector(m1_y) == repr_vector(m1.mul_vector(x_m1))
        && repr_vector(m2_y) == repr_vector(m2.mul_vector(x_m2))
        && repr_vector(m3_y) == repr_vector(m3.mul_vector(x_m3)))
        std::cout << "YES (Matrices for " << name << " multiplying by vector (sequential) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " multiplying by vector (sequential) are invalid)\n";
}

template <typename MatrixType>
void matr_scalar(const MatrixType& m1, const MatrixType& m2, const MatrixType& m3, const std::string name) {
    std::string m1_ =
        "0.00 0.00 16.70 0.00 0.00 0.00 0.00 10.02 0.00";

    std::string m2_ =
        "0.00 0.00 0.00 0.00 13.36 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
        "23.38 0.00 0.00 0.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 6.68 0.00 0.00 0.00\n"
        "0.00 0.00 30.06 0.00 0.00 0.00 3.34";

    std::string m3_ =
        "0.00 0.00 0.00 26.72\n"
        "0.00 10.02 0.00 0.00\n"
        "16.70 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00\n"
        "0.00 23.38 0.00 3.34\n"
        "0.00 0.00 13.36 0.00";

    double alpha = 3.34;

    if (m1_ == m1.mul_scalar_p(alpha).repr() && m2_ == m2.mul_scalar_p(alpha).repr() && m3_ == m3.mul_scalar_p(alpha).repr())
        std::cout << "YES (Matrices for " << name << " multiplying by scalar (parallel) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " multiplying by scalar (parallel) are invalid)\n";

    if (m1_ == m1.mul_scalar(alpha).repr() && m2_ == m2.mul_scalar(alpha).repr() && m3_ == m3.mul_scalar(alpha).repr())
        std::cout << "YES (Matrices for " << name << " multiplying by scalar (sequential) are valid)\n";
    else std::cout << "NO (Matrices for " << name << " multiplying by scalar (sequential) are invalid)\n";
}

template <typename MatrixType>
void matr_repr(const MatrixType& m1, const MatrixType& m2, const MatrixType& m3, const std::string name) {
    std::string m1_ =
        "0.00 0.00 5.00 0.00 0.00 0.00 0.00 3.00 0.00";

    std::string m2_ =
        "0.00 0.00 0.00 0.00 4.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
        "7.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 2.00 0.00 0.00 0.00\n"
        "0.00 0.00 9.00 0.00 0.00 0.00 1.00";

    std::string m3_ =
        "0.00 0.00 0.00 8.00\n"
        "0.00 3.00 0.00 0.00\n"
        "5.00 0.00 0.00 0.00\n"
        "0.00 0.00 0.00 0.00\n"
        "0.00 7.00 0.00 1.00\n"
        "0.00 0.00 4.00 0.00";

    if (m1_ == m1.repr() && m2_ == m2.repr() && m3_ == m3.repr())
        std::cout << "YES (Matrices for " << name << " are valid)\n";
    else std::cout << "NO (Matrices for " << name << " are invalid)\n";
}

int main() {
    std::string m1 =
        "0 0 5 0 0 0 0 3 0";

    std::string m2 =
        "0 0 0 0 4 0 0\n"
        "0 0 0 0 0 0 0\n"
        "7 0 0 0 0 0 0\n"
        "0 0 0 2 0 0 0\n"
        "0 0 9 0 0 0 1";

    std::string m3 =
        "0 0 0 8\n"
        "0 3 0 0\n"
        "5 0 0 0\n"
        "0 0 0 0\n"
        "0 7 0 1\n"
        "0 0 4 0";

    DenseMatrix m1_d(1, 9);
    double m1d[1][9] = {
        {0, 0, 5, 0, 0, 0, 0, 3, 0}
    };
    for (int r = 0; r < 1; r++)
        for (int c = 0; c < 9; c++)
            m1_d(r, c) = m1d[r][c];

    DenseMatrix m2_d(5, 7);
    double m2d[5][7] = {
        {0, 0, 0, 0, 4, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {7, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 2, 0, 0, 0},
        {0, 0, 9, 0, 0, 0, 1}
    };
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 7; c++)
            m2_d(r, c) = m2d[r][c];

    DenseMatrix m3_d(6, 4);
    double m3d[6][4] = {
        {0, 0, 0, 8},
        {0, 3, 0, 0},
        {5, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 7, 0, 1},
        {0, 0, 4, 0}
    };
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 4; c++)
            m3_d(r, c) = m3d[r][c];

    COOMatrix m1_coo(1, 9);
    m1_coo.add_entry(0, 2, 5);
    m1_coo.add_entry(0, 7, 3);

    COOMatrix m2_coo(5, 7);
    m2_coo.add_entry(0, 4, 4);
    m2_coo.add_entry(2, 0, 7);
    m2_coo.add_entry(3, 3, 2);
    m2_coo.add_entry(4, 2, 9);
    m2_coo.add_entry(4, 6, 1);

    COOMatrix m3_coo(6, 4);
    m3_coo.add_entry(0, 3, 8);
    m3_coo.add_entry(1, 1, 3);
    m3_coo.add_entry(2, 0, 5);
    m3_coo.add_entry(4, 1, 7);
    m3_coo.add_entry(4, 3, 1);
    m3_coo.add_entry(5, 2, 4);

    CSRMatrix m1_csr(1, 9);
    m1_csr.add_entry(0, 2, 5);
    m1_csr.add_entry(0, 7, 3);
    m1_csr.end_creation();

    CSRMatrix m2_csr(5, 7);
    m2_csr.add_entry(0, 4, 4);
    m2_csr.add_entry(2, 0, 7);
    m2_csr.add_entry(3, 3, 2);
    m2_csr.add_entry(4, 2, 9);
    m2_csr.add_entry(4, 6, 1);
    m2_csr.end_creation();

    CSRMatrix m3_csr(6, 4);
    m3_csr.add_entry(0, 3, 8);
    m3_csr.add_entry(1, 1, 3);
    m3_csr.add_entry(2, 0, 5);
    m3_csr.add_entry(4, 1, 7);
    m3_csr.add_entry(4, 3, 1);
    m3_csr.add_entry(5, 2, 4);
    m3_csr.end_creation();

    CSCMatrix m1_csc(1, 9);
    m1_csc.add_entry(0, 2, 5);
    m1_csc.add_entry(0, 7, 3);
    m1_csc.end_creation();

    CSCMatrix m2_csc(5, 7);
    m2_csc.add_entry(2, 0, 7);
    m2_csc.add_entry(4, 2, 9);
    m2_csc.add_entry(3, 3, 2);
    m2_csc.add_entry(0, 4, 4);
    m2_csc.add_entry(4, 6, 1);
    m2_csc.end_creation();

    CSCMatrix m3_csc(6, 4);
    m3_csc.add_entry(2, 0, 5);
    m3_csc.add_entry(1, 1, 3);
    m3_csc.add_entry(4, 1, 7);
    m3_csc.add_entry(5, 2, 4);
    m3_csc.add_entry(0, 3, 8);
    m3_csc.add_entry(4, 3, 1);
    m3_csc.end_creation();

    // Правильность матриц
    matr_repr(m1_d, m2_d, m3_d, "Dense");
    matr_repr(m1_coo, m2_coo, m3_coo, "COO");
    matr_repr(m1_csr, m2_csr, m3_csr, "CSR");
    matr_repr(m1_csc, m2_csc, m3_csc, "CSC");

    // Умножение на скаляр
    matr_scalar(m1_d, m2_d, m3_d, "Dense");
    matr_scalar(m1_coo, m2_coo, m3_coo, "COO");
    matr_scalar(m1_csr, m2_csr, m3_csr, "CSR");
    matr_scalar(m1_csc, m2_csc, m3_csc, "CSC");

    // Умножение на вектор
    matr_vec(m1_d, m2_d, m3_d, "Dense");
    matr_vec(m1_coo, m2_coo, m3_coo, "COO");
    matr_vec(m1_csr, m2_csr, m3_csr, "CSR");
    matr_vec(m1_csc, m2_csc, m3_csc, "CSC");

    // Транспонирование
    matr_transpose(m1_d, m2_d, m3_d, "Dense");
    matr_transpose(m1_coo, m2_coo, m3_coo, "COO");
    matr_transpose(m1_csr, m2_csr, m3_csr, "CSR");
    matr_transpose(m1_csc, m2_csc, m3_csc, "CSC");

    DenseMatrix m11_d(1, 9);
    m11_d(0, 0) = 1.0;
    m11_d(0, 1) = 0.0;
    m11_d(0, 2) = 0.0;
    m11_d(0, 3) = 0.0;
    m11_d(0, 4) = 2.0;
    m11_d(0, 5) = 0.0;
    m11_d(0, 6) = 0.0;
    m11_d(0, 7) = 0.0;
    m11_d(0, 8) = 4.0;

    COOMatrix m11_coo(1, 9);
    m11_coo.add_entry(0, 0, 1.0);
    m11_coo.add_entry(0, 4, 2.0);
    m11_coo.add_entry(0, 8, 4.0);

    CSRMatrix m11_csr(1, 9);
    m11_csr.add_entry(0, 0, 1.0);
    m11_csr.add_entry(0, 4, 2.0);
    m11_csr.add_entry(0, 8, 4.0);
    m11_csr.end_creation();

    CSCMatrix m11_csc(1, 9);
    m11_csc.add_entry(0, 0, 1.0);
    m11_csc.add_entry(0, 4, 2.0);
    m11_csc.add_entry(0, 8, 4.0);
    m11_csc.end_creation();

    DenseMatrix m22_d(5, 7);
    m22_d(0, 3) = 1.0;
    m22_d(1, 2) = 2.0;
    m22_d(2, 5) = 3.0;
    m22_d(3, 6) = 4.0;
    m22_d(4, 0) = 5.0;

    COOMatrix m22_coo(5, 7);
    m22_coo.add_entry(0, 3, 1.0);
    m22_coo.add_entry(1, 2, 2.0);
    m22_coo.add_entry(2, 5, 3.0);
    m22_coo.add_entry(3, 6, 4.0);
    m22_coo.add_entry(4, 0, 5.0);

    CSRMatrix m22_csr(5, 7);
    m22_csr.add_entry(0, 3, 1.0);
    m22_csr.add_entry(1, 2, 2.0);
    m22_csr.add_entry(2, 5, 3.0);
    m22_csr.add_entry(3, 6, 4.0);
    m22_csr.add_entry(4, 0, 5.0);
    m22_csr.end_creation();

    CSCMatrix m22_csc(5, 7);
    m22_csc.add_entry(4, 0, 5.0);
    m22_csc.add_entry(1, 2, 2.0);
    m22_csc.add_entry(0, 3, 1.0);
    m22_csc.add_entry(2, 5, 3.0);
    m22_csc.add_entry(3, 6, 4.0);
    m22_csc.end_creation();

    DenseMatrix m33_d(6, 4);
    m33_d(0, 1) = 1.0;
    m33_d(1, 3) = 2.0;
    m33_d(2, 2) = 3.0;
    m33_d(3, 0) = 4.0;
    m33_d(5, 1) = 5.0;

    COOMatrix m33_coo(6, 4);
    m33_coo.add_entry(0, 1, 1.0);
    m33_coo.add_entry(1, 3, 2.0);
    m33_coo.add_entry(2, 2, 3.0);
    m33_coo.add_entry(3, 0, 4.0);
    m33_coo.add_entry(5, 1, 5.0);

    CSRMatrix m33_csr(6, 4);
    m33_csr.add_entry(0, 1, 1.0);
    m33_csr.add_entry(1, 3, 2.0);
    m33_csr.add_entry(2, 2, 3.0);
    m33_csr.add_entry(3, 0, 4.0);
    m33_csr.add_entry(5, 1, 5.0);
    m33_csr.end_creation();

    CSCMatrix m33_csc(6, 4);
    m33_csc.add_entry(3, 0, 4.0);
    m33_csc.add_entry(0, 1, 1.0);
    m33_csc.add_entry(5, 1, 5.0);
    m33_csc.add_entry(2, 2, 3.0);
    m33_csc.add_entry(1, 3, 2.0);
    m33_csc.end_creation();

    // Сложение матриц
    matr_add(m1_d, m11_d, m2_d, m22_d, m3_d, m33_d, "Dense");
    matr_add(m1_coo, m11_coo, m2_coo, m22_coo, m3_coo, m33_coo, "COO");
    matr_add(m1_csr, m11_csr, m2_csr, m22_csr, m3_csr, m33_csr, "CSR");
    matr_add(m1_csc, m11_csc, m2_csc, m22_csc, m3_csc, m33_csc, "CSC");

    return 0;
}
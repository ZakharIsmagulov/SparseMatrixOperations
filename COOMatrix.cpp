#include "COOMatrix.h"
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <string>
#include "VecRepr.h"

// Для сортировки
struct COOEntry {
    int r, c;
    double v;
};

COOMatrix::COOMatrix(int rows, int cols)
    : rows_(rows), cols_(cols)
{}

void COOMatrix::add_entry(int r, int c, double v) {
    row_idx_.push_back(r);
    col_idx_.push_back(c);
    val_.push_back(v);
}

COOMatrix::COOMatrix(const std::string& filename) {
    FILE* f = std::fopen(filename.c_str(), "r");

    int r, c, nz;
    std::fscanf(f, "%d %d %d", &r, &c, &nz);
    rows_ = r;
    cols_ = c;
    row_idx_.reserve(nz);
    col_idx_.reserve(nz);
    val_.reserve(nz);

    for (int k = 0; k < nz; ++k) {
        int i, j;
        double v;
        std::fscanf(f, "%d %d %lf", &i, &j, &v);
        add_entry(i, j, v);
    }

    std::fclose(f);
}

COOMatrix COOMatrix::mul_scalar_p(double alpha) const {
    COOMatrix res(rows_, cols_);
    int n = nnz();

    res.row_idx_ = row_idx_;
    res.col_idx_ = col_idx_;
    res.val_.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> COOMatrix::mul_vector_p(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);
    int n = nnz();

    // Атомик нужен для того, чтобы лочить res[r], пока транзакция выполняется
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int r = row_idx_[i];
        int c = col_idx_[i];
        double a = val_[i];

        #pragma omp atomic
        res[r] += a * x[c];
    }

    return res;
}

COOMatrix COOMatrix::transpose_p() const {
    COOMatrix res(cols_, rows_);
    int n = nnz();

    res.row_idx_.resize(n);
    res.col_idx_.resize(n);
    res.val_.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.row_idx_[i] = col_idx_[i];
        res.col_idx_[i] = row_idx_[i];
        res.val_[i] = val_[i];
    }
    return res;
}

COOMatrix COOMatrix::add_p(const COOMatrix& other) const {
    int n1 = nnz();
    int n2 = other.nnz();

    // Сливаем и сортируем координаты
    std::vector<COOEntry> all(n1 + n2);

    #pragma omp parallel for
    for (int i = 0; i < n1; i++) {
        all[i] = { row_idx_[i], col_idx_[i], val_[i] };
    }

    #pragma omp parallel for
    for (int i = 0; i < n2; i++) {
        all[n1 + i] = { other.row_idx_[i], other.col_idx_[i], other.val_[i] };
    }

    std::sort(all.begin(), all.end(), [](const COOEntry& a, const COOEntry& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
        });

    // Одинаковые r,c складываем
    COOMatrix res(rows_, cols_);
    int m = (int)all.size();

    int i = 0;
    while (i < m) {
        int r = all[i].r;
        int c = all[i].c;
        double s = all[i].v;
        i++;

        if (i < m && all[i].r == r && all[i].c == c) {
            s += all[i].v;
            i++;
        }

        if (s != 0.0) {
            res.add_entry(r, c, s);
        }
    }

    return res;
}

COOMatrix COOMatrix::mul_scalar(double alpha) const {
    COOMatrix res(rows_, cols_);
    int n = nnz();

    res.row_idx_ = row_idx_;
    res.col_idx_ = col_idx_;
    res.val_.resize(n);

    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> COOMatrix::mul_vector(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);
    int n = nnz();

    for (int i = 0; i < n; i++) {
        int r = row_idx_[i];
        int c = col_idx_[i];
        double a = val_[i];
        res[r] += a * x[c];
    }

    return res;
}

COOMatrix COOMatrix::transpose() const {
    COOMatrix res(cols_, rows_);
    int n = nnz();

    res.row_idx_.resize(n);
    res.col_idx_.resize(n);
    res.val_.resize(n);

    for (int i = 0; i < n; i++) {
        res.row_idx_[i] = col_idx_[i];
        res.col_idx_[i] = row_idx_[i];
        res.val_[i] = val_[i];
    }
    return res;
}

COOMatrix COOMatrix::add(const COOMatrix& other) const {
    int n1 = nnz();
    int n2 = other.nnz();

    // Сливаем и сортируем координаты
    std::vector<COOEntry> all(n1 + n2);

    for (int i = 0; i < n1; i++) {
        all[i] = { row_idx_[i], col_idx_[i], val_[i] };
    }

    for (int i = 0; i < n2; i++) {
        all[n1 + i] = { other.row_idx_[i], other.col_idx_[i], other.val_[i] };
    }

    std::sort(all.begin(), all.end(), [](const COOEntry& a, const COOEntry& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
        });

    // Одинаковые r,c складываем
    COOMatrix res(rows_, cols_);
    int m = (int)all.size();

    int i = 0;
    while (i < m) {
        int r = all[i].r;
        int c = all[i].c;
        double s = all[i].v;
        i++;

        if (i < m && all[i].r == r && all[i].c == c) {
            s += all[i].v;
            i++;
        }

        if (s != 0.0) {
            res.add_entry(r, c, s);
        }
    }

    return res;
}

void COOMatrix::print_matrix() const {
    std::vector<double> dense(rows_ * cols_, 0.0);
    int n = nnz();
    for (int k = 0; k < n; k++) {
        dense[row_idx_[k] * cols_ + col_idx_[k]] = val_[k];
    }

    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            std::cout << dense[r * cols_ + c] << " ";
        }
        std::cout << "\n";
    }
}

std::string COOMatrix::repr() const {
    std::vector<double> dense(rows_ * cols_, 0.0);

    for (int k = 0; k < nnz(); ++k)
        dense[row_idx_[k] * cols_ + col_idx_[k]] = val_[k];

    std::string s;

    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            s += format2(dense[r * cols_ + c]);
            if (c + 1 < cols_)
                s += " ";
        }
        if (r + 1 < rows_)
            s += "\n";
    }

    return s;
}

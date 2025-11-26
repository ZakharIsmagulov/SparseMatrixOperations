#include "CSCMatrix.h"
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <string>
#include "VecRepr.h"

// Для сортировки после транспонирования и конвертирования из COO
struct CSCEntry {
    int r;
    double v;
};

CSCMatrix::CSCMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), col_ptr_(cols + 1, 0), last_col_(0), last_row_(-1)
{}

CSCMatrix::CSCMatrix(const COOMatrix& coo){
    rows_ = coo.rows();
    cols_ = coo.cols();
    last_row_ = -1;
    last_col_ = 0;
    const std::vector<int>& r_idx = coo.row_idx();
    const std::vector<int>& c_idx = coo.col_idx();
    const std::vector<double>& v_val = coo.val();
    int nz = coo.nnz();

    col_ptr_ = std::vector<int>(cols_ + 1, 0);
    row_idx_.resize(nz);
    val_.resize(nz);

    // Количество ненулевых в каждом столбце
    for (int c : c_idx) {
        col_ptr_[c]++;
    }

    int sum = 0;
    for (int c = 0; c < cols_; c++) {
        int cnt = col_ptr_[c];
        col_ptr_[c] = sum;
        sum += cnt;
    }
    col_ptr_[cols_] = sum;

    // offset[c] — текущая позиция для столбца c
    std::vector<int> offset(cols_);
    for (int c = 0; c < cols_; c++) {
        offset[c] = col_ptr_[c];
    }

    for (int k = 0; k < nz; k++) {
        int c = c_idx[k];
        int pos = offset[c]++;

        row_idx_[pos] = r_idx[k];
        val_[pos] = v_val[k];
    }

    // Так как в COO не гарантирован порядок
    for (int c = 0; c < cols_; c++) {
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];

        std::vector<CSCEntry> tmp;
        tmp.reserve(end - start);
        for (int k = start; k < end; k++) {
            tmp.push_back({ row_idx_[k], val_[k] });
        }

        std::sort(tmp.begin(), tmp.end(),
            [](const CSCEntry& a, const CSCEntry& b) { return a.r < b.r; });

        for (int k = start, i = 0; k < end; k++, i++) {
            row_idx_[k] = tmp[i].r;
            val_[k] = tmp[i].v;
        }
    }

    last_col_ = cols_;
}

void CSCMatrix::add_entry(int r, int c, double v) {
    // Если нарушен порядок добавления
    if ((last_col_ > c) or (last_col_ == c && last_row_ >= r)) {
        std::cout << "add_entry was ignored because of invalid order of adding elements";
        return;
    }

    while (last_col_ < c) {
        col_ptr_[last_col_ + 1] = (int)val_.size();
        last_col_++;
    }

    row_idx_.push_back(r);
    val_.push_back(v);
    last_row_ = r;
}

void CSCMatrix::end_creation() {
    while (last_col_ < cols_) {
        col_ptr_[last_col_ + 1] = (int)val_.size();
        last_col_++;
    }
}

CSCMatrix CSCMatrix::mul_scalar_p(double alpha) const {
    CSCMatrix res(rows_, cols_);
    int n = nnz();

    res.col_ptr_ = col_ptr_;
    res.row_idx_ = row_idx_;
    res.val_.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> CSCMatrix::mul_vector_p(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    // atomic нужен, т.к. параллелизм по столбцам => задействованы одни и те же y[r]
    #pragma omp parallel for
    for (int c = 0; c < cols_; c++) {
        double xc = x[c];
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];
        for (int k = start; k < end; k++) {
            int r = row_idx_[k];
            double a = val_[k];

            #pragma omp atomic
            res[r] += a * xc;
        }
    }

    return res;
}

CSCMatrix CSCMatrix::transpose_p() const {
    int n = nnz();
    CSCMatrix res(cols_, rows_);

    // Количество элементов в каждой строке
    std::vector<int> row_count(rows_, 0);

    #pragma omp parallel
    {
        // Счётчики для каждого потока
        std::vector<int> local(rows_, 0);

        // nowait, т.к. далее будет гонка данных за row_count => лучше, чтобы потоки завершали этот цикл асинхронно
        #pragma omp for nowait
        for (int c = 0; c < cols_; c++) {
            int start = col_ptr_[c];
            int end = col_ptr_[c + 1];
            for (int k = start; k < end; k++) {
                local[row_idx_[k]]++;
            }
        }

        // В эту секцию одновременно может попасть только один поток
        #pragma omp critical
        {
            for (int r = 0; r < rows_; r++) {
                row_count[r] += local[r];
            }
        }
    }

    // Создание новых значений столбцов
    res.col_ptr_[0] = 0;
    for (int r = 0; r < rows_; r++) {
        res.col_ptr_[r + 1] = res.col_ptr_[r] + row_count[r];
    }

    res.row_idx_.resize(n);
    res.val_.resize(n);

    // offset[r] - следующая свободная позиция в столбце r транспонированной матрицы
    std::vector<int> offset = res.col_ptr_;

    // Параллельная запись столбцов в новую матрицу по строкам предыдущей по порядку
    #pragma omp parallel for
    for (int c = 0; c < cols_; c++) {
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];

        for (int k = start; k < end; k++) {
            int r = row_idx_[k];

            int pos;
            // atomic, т.к. только один поток должен работать с offset[r], capture, т.к. 2 операции: ++ и = в транзакции
            #pragma omp atomic capture
            pos = offset[r]++;

            res.row_idx_[pos] = c;
            res.val_[pos] = val_[k];
        }
    }

    // Т.к. индексы строк могли перемешаться из-за параллельности, то надо их отсортировать
    #pragma omp parallel for
    for (int c = 0; c < res.cols_; c++) {
        int start = res.col_ptr_[c];
        int end = res.col_ptr_[c + 1];

        std::vector<CSCEntry> tmp;
        tmp.reserve(end - start);
        for (int k = start; k < end; k++) {
            tmp.push_back({ res.row_idx_[k], res.val_[k] });
        }

        std::sort(tmp.begin(), tmp.end(),
            [](const CSCEntry& a, const CSCEntry& b) { return a.r < b.r; });

        for (int k = start, i = 0; k < end; k++, i++) {
            res.row_idx_[k] = tmp[i].r;
            res.val_[k] = tmp[i].v;
        }
    }

    res.last_col_ = res.cols_;

    return res;
}

CSCMatrix CSCMatrix::add_p(const CSCMatrix& other) const {
    CSCMatrix res(rows_, cols_);

    // Количество ненулевых элементов в новой матрице в каждом столбце
    std::vector<int> col_nnz(cols_, 0);

    #pragma omp parallel for
    for (int c = 0; c < cols_; c++) {
        int a = col_ptr_[c], a_end = col_ptr_[c + 1];
        int b = other.col_ptr_[c], b_end = other.col_ptr_[c + 1];

        int count = 0;
        while (a < a_end || b < b_end) {
            if (a >= a_end || b >= b_end) {
                count++;
                a++;
                b++;
                continue;
            }
            if (a < a_end && row_idx_[a] < other.row_idx_[b]) {
                count++;
                a++;
            }
            else if (b < b_end && other.row_idx_[b] < row_idx_[a]) {
                count++;
                b++;
            }
            else {
                if (val_[a] + other.val_[b] != 0.0) count++;
                a++;
                b++;
            }
        }
        col_nnz[c] = count;
    }

    // Создание новых значений столбцов
    res.col_ptr_[0] = 0;
    for (int c = 0; c < cols_; c++) {
        res.col_ptr_[c + 1] = res.col_ptr_[c] + col_nnz[c];
    }

    int total = res.col_ptr_[cols_];
    res.row_idx_.resize(total);
    res.val_.resize(total);

    // Параллельная запись столбцов в новую матрицу
    #pragma omp parallel for
    for (int c = 0; c < cols_; c++) {
        int a = col_ptr_[c], a_end = col_ptr_[c + 1];
        int b = other.col_ptr_[c], b_end = other.col_ptr_[c + 1];

        int pos = res.col_ptr_[c];

        while (a < a_end || b < b_end) {
            if (a >= a_end) {
                res.row_idx_[pos] = other.row_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
                continue;
            }
            if (b >= b_end) {
                res.row_idx_[pos] = row_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
                continue;
            }
            if (a < a_end && row_idx_[a] < other.row_idx_[b]) {
                res.row_idx_[pos] = row_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
            }
            else if (b < b_end && other.row_idx_[b] < row_idx_[a]) {
                res.row_idx_[pos] = other.row_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
            }
            else {
                double s = val_[a] + other.val_[b];
                if (s != 0.0) {
                    res.row_idx_[pos] = row_idx_[a];
                    res.val_[pos] = s;
                    pos++;
                }
                a++;
                b++;
            }
        }
    }

    res.last_col_ = res.cols_;
    return res;
}

CSCMatrix CSCMatrix::mul_scalar(double alpha) const {
    CSCMatrix res(rows_, cols_);
    int n = nnz();

    res.col_ptr_ = col_ptr_;
    res.row_idx_ = row_idx_;
    res.val_.resize(n);

    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> CSCMatrix::mul_vector(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    for (int c = 0; c < cols_; c++) {
        double xc = x[c];
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];
        for (int k = start; k < end; k++) {
            int r = row_idx_[k];
            double a = val_[k];
            res[r] += a * xc;
        }
    }

    return res;
}

CSCMatrix CSCMatrix::transpose() const {
    int n = nnz();
    CSCMatrix res(cols_, rows_);

    // Количество элементов в каждой строке
    std::vector<int> row_count(rows_, 0);
    for (int c = 0; c < cols_; c++) {
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];
        for (int k = start; k < end; k++) {
            row_count[row_idx_[k]]++;
        }
    }

    // Создание новых значений столбцов
    res.col_ptr_[0] = 0;
    for (int r = 0; r < rows_; r++) {
        res.col_ptr_[r + 1] = res.col_ptr_[r] + row_count[r];
    }

    res.row_idx_.resize(n);
    res.val_.resize(n);

    // offset[r] - следующая свободная позиция в столбце r транспонированной матрицы
    std::vector<int> offset = res.col_ptr_;

    // Последовательная запись столбцов в новую матрицу по строкам предыдущей по порядку
    for (int c = 0; c < cols_; c++) {
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];

        for (int k = start; k < end; k++) {
            int r = row_idx_[k];

            int pos;
            pos = offset[r]++;
            res.row_idx_[pos] = c;
            res.val_[pos] = val_[k];
        }
    }

    res.last_col_ = res.cols_;

    return res;
}

CSCMatrix CSCMatrix::add(const CSCMatrix& other) const {
    CSCMatrix res(rows_, cols_);

    // Количество ненулевых элементов в новой матрице в каждом столбце
    std::vector<int> col_nnz(cols_, 0);

    for (int c = 0; c < cols_; c++) {
        int a = col_ptr_[c], a_end = col_ptr_[c + 1];
        int b = other.col_ptr_[c], b_end = other.col_ptr_[c + 1];

        int count = 0;
        while (a < a_end || b < b_end) {
            if (a >= a_end || b >= b_end) {
                count++;
                a++;
                b++;
                continue;
            }
            if (a < a_end && row_idx_[a] < other.row_idx_[b]) {
                count++;
                a++;
            }
            else if (b < b_end && other.row_idx_[b] < row_idx_[a]) {
                count++;
                b++;
            }
            else {
                if (val_[a] + other.val_[b] != 0.0) count++;
                a++;
                b++;
            }
        }
        col_nnz[c] = count;
    }

    // Создание новых значений столбцов
    res.col_ptr_[0] = 0;
    for (int c = 0; c < cols_; c++) {
        res.col_ptr_[c + 1] = res.col_ptr_[c] + col_nnz[c];
    }

    int total = res.col_ptr_[cols_];
    res.row_idx_.resize(total);
    res.val_.resize(total);

    // Последовательная запись столбцов в новую матрицу
    for (int c = 0; c < cols_; c++) {
        int a = col_ptr_[c], a_end = col_ptr_[c + 1];
        int b = other.col_ptr_[c], b_end = other.col_ptr_[c + 1];

        int pos = res.col_ptr_[c];

        while (a < a_end || b < b_end) {
            if (a >= a_end) {
                res.row_idx_[pos] = other.row_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
                continue;
            }
            if (b >= b_end) {
                res.row_idx_[pos] = row_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
                continue;
            }
            if (a < a_end && row_idx_[a] < other.row_idx_[b]) {
                res.row_idx_[pos] = row_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
            }
            else if (b < b_end && other.row_idx_[b] < row_idx_[a]) {
                res.row_idx_[pos] = other.row_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
            }
            else {
                double s = val_[a] + other.val_[b];
                if (s != 0.0) {
                    res.row_idx_[pos] = row_idx_[a];
                    res.val_[pos] = s;
                    pos++;
                }
                a++;
                b++;
            }
        }
    }

    res.last_col_ = res.cols_;
    return res;
}

void CSCMatrix::print_matrix() const {
    std::vector<double> dense(rows_ * cols_, 0.0);
    int n = nnz();

    for (int c = 0; c < cols_; c++) {
        int start = col_ptr_[c];
        int end = col_ptr_[c + 1];
        for (int k = start; k < end; k++) {
            int r = row_idx_[k];
            dense[r * cols_ + c] = val_[k];
        }
    }

    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            std::cout << dense[r * cols_ + c] << " ";
        }
        std::cout << "\n";
    }
}

std::string CSCMatrix::repr() const {
    std::vector<double> dense(rows_ * cols_, 0.0);

    for (int c = 0; c < cols_; c++)
        for (int idx = col_ptr_[c]; idx < col_ptr_[c + 1]; ++idx)
            dense[row_idx_[idx] * cols_ + c] = val_[idx];

    std::string s;

    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            s += format2(dense[r * cols_ + c]);
            if (c + 1 < cols_)
                s += " ";
        }
        if (r + 1 < rows_)
            s += "\n";
    }

    return s;
}

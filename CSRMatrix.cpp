#include "CSRMatrix.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <string>
#include "VecRepr.h"

// Для сортировки после транспонирования или конвертирования из COO
struct CSREntry {
    int c;
    double v;
};

CSRMatrix::CSRMatrix(int rows, int cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0), last_row_(0), last_col_(-1)
{}

void CSRMatrix::add_entry(int r, int c, double v) {
    // Если нарушен порядок добавления
    if ((last_row_ > r) or (last_row_ == r && last_col_ >= c)) {
        std::cout << "add_entry was ignored because of invalid order of adding elements";
        return;
    }

    while (last_row_ < r) {
        row_ptr_[last_row_ + 1] = (int)val_.size();
        last_row_++;
    }

    col_idx_.push_back(c);
    val_.push_back(v);
    last_col_ = c;
}

CSRMatrix::CSRMatrix(const COOMatrix& coo) {
    rows_ = coo.rows();
    cols_ = coo.cols();
    last_row_ = 0;
    last_col_ = -1;
    const std::vector<int>& r_idx = coo.row_idx();
    const std::vector<int>& c_idx = coo.col_idx();
    const std::vector<double>& v_val = coo.val();
    int nz = coo.nnz();

    row_ptr_ = std::vector<int>(rows_ + 1, 0);
    col_idx_.resize(nz);
    val_.resize(nz);

    // Количество ненулевых в каждой строке
    for (int r : r_idx) {
        row_ptr_[r]++;
    }

    int sum = 0;
    for (int r = 0; r < rows_; r++) {
        int cnt = row_ptr_[r];
        row_ptr_[r] = sum;
        sum += cnt;
    }
    row_ptr_[rows_] = sum;

    // offset[r] — текущая позиция для строки r
    std::vector<int> offset(rows_);
    for (int r = 0; r < rows_; r++) {
        offset[r] = row_ptr_[r];
    }

    for (int k = 0; k < nz; k++) {
        int r = r_idx[k];
        int pos = offset[r]++;

        col_idx_[pos] = c_idx[k];
        val_[pos] = v_val[k];
    }

    // Так как в COO не гарантирован порядок
    for (int r = 0; r < rows_; r++) {
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];

        std::vector<CSREntry> tmp;
        tmp.reserve(end - start);
        for (int k = start; k < end; k++) {
            tmp.push_back({ col_idx_[k], val_[k] });
        }

        std::sort(tmp.begin(), tmp.end(),
            [](const CSREntry& a, const CSREntry& b) { return a.c < b.c; });

        for (int k = start, i = 0; k < end; k++, i++) {
            col_idx_[k] = tmp[i].c;
            val_[k] = tmp[i].v;
        }
    }

    last_row_ = rows_;
}

void CSRMatrix::end_creation() {
    while (last_row_ < rows_) {
        row_ptr_[last_row_ + 1] = (int)val_.size();
        last_row_++;
    }
}

CSRMatrix CSRMatrix::mul_scalar_p(double alpha) const {
    CSRMatrix res(rows_, cols_);
    int n = nnz();

    res.row_ptr_ = row_ptr_;
    res.col_idx_ = col_idx_;
    res.val_.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> CSRMatrix::mul_vector_p(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        double sum = 0.0;
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];
        for (int k = start; k < end; k++) {
            sum += val_[k] * x[col_idx_[k]];
        }
        res[r] = sum;
    }
    return res;
}

CSRMatrix CSRMatrix::transpose_p() const {
    int n = nnz();
    CSRMatrix res(cols_, rows_);

    // Количество элементов в каждом столбце
    std::vector<int> col_count(cols_, 0);

    #pragma omp parallel
    {
        // Счётчики для каждого потока
        std::vector<int> local(cols_, 0);

        // nowait, т.к. далее будет гонка данных за col_count => лучше, чтобы потоки завершали этот цикл асинхронно
        #pragma omp for nowait
        for (int r = 0; r < rows_; r++) {
            int start = row_ptr_[r];
            int end = row_ptr_[r + 1];
            for (int k = start; k < end; k++) {
                local[col_idx_[k]]++;
            }
        }
           
        // В эту секцию одновременно может попасть только один поток
        #pragma omp critical
        {
            for (int c = 0; c < cols_; c++) {
                col_count[c] += local[c];
            }
        }
    }

    // Создание новых значений строк
    res.row_ptr_[0] = 0;
    for (int c = 0; c < cols_; c++) {
        res.row_ptr_[c + 1] = res.row_ptr_[c] + col_count[c];
    }

    res.col_idx_.resize(n);
    res.val_.resize(n);

    // offset[c] - следующая свободная позиция в строке c транспонированной матрицы
    std::vector<int> offset = res.row_ptr_;

    // Параллельная запись строк в новую матрицу по столбцам предыдущей по порядку
    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];

        for (int k = start; k < end; k++) {
            int c = col_idx_[k];

            int pos;

            // atomic, т.к. только один поток должен работать с offset[c], capture, т.к. 2 операции: ++ и = в транзакции
            #pragma omp atomic capture
            pos = offset[c]++;

            res.col_idx_[pos] = r;
            res.val_[pos] = val_[k];
        }
    }

    // Т.к. индексы столбцов могли перемешаться из-за параллельности, то надо их отсортировать
    #pragma omp parallel for
    for (int r = 0; r < res.rows_; r++) {
        int start = res.row_ptr_[r];
        int end = res.row_ptr_[r + 1];

        std::vector<CSREntry> tmp;
        tmp.reserve(end - start);
        for (int k = start; k < end; k++) {
            tmp.push_back({ res.col_idx_[k], res.val_[k] });
        }

        std::sort(tmp.begin(), tmp.end(),
            [](const CSREntry& a, const CSREntry& b) { return a.c < b.c; });

        for (int k = start, i = 0; k < end; k++, i++) {
            res.col_idx_[k] = tmp[i].c;
            res.val_[k] = tmp[i].v;
        }
    }

    res.last_row_ = res.rows_;
    return res;
}

CSRMatrix CSRMatrix::add_p(const CSRMatrix& other) const {
    CSRMatrix res(rows_, cols_);

    // Количество ненулевых элементов в новой матрице в каждой строке
    std::vector<int> row_nnz(rows_, 0);

    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        int a = row_ptr_[r], a_end = row_ptr_[r + 1];
        int b = other.row_ptr_[r], b_end = other.row_ptr_[r + 1];

        int count = 0;
        while (a < a_end || b < b_end) {
            if (a >= a_end || b >= b_end) {
                count++;
                a++;
                b++;
                continue;
            }
            if (a < a_end && col_idx_[a] < other.col_idx_[b]) {
                count++;
                a++;
            }
            else if (b < b_end && other.col_idx_[b] < col_idx_[a]) {
                count++;
                b++;
            }
            else {
                if (val_[a] + other.val_[b] != 0.0) count++;
                a++;
                b++;
            }
        }
        row_nnz[r] = count;
    }

    // Создание новых значений строк
    res.row_ptr_[0] = 0;
    for (int r = 0; r < rows_; r++) {
        res.row_ptr_[r + 1] = res.row_ptr_[r] + row_nnz[r];
    }

    int total = res.row_ptr_[rows_];
    res.col_idx_.resize(total);
    res.val_.resize(total);

    // Параллельная запись строк в новую матрицу
    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        int a = row_ptr_[r], a_end = row_ptr_[r + 1];
        int b = other.row_ptr_[r], b_end = other.row_ptr_[r + 1];

        int pos = res.row_ptr_[r];

        while (a < a_end || b < b_end) {
            if (a >= a_end) {
                res.col_idx_[pos] = other.col_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
                continue;
            }
            if (b >= b_end) {
                res.col_idx_[pos] = col_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
                continue;
            }
            if (a < a_end && col_idx_[a] < other.col_idx_[b]) {
                res.col_idx_[pos] = col_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
            }
            else if (b < b_end && other.col_idx_[b] < col_idx_[a]) {
                res.col_idx_[pos] = other.col_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
            }
            else {
                double s = val_[a] + other.val_[b];
                if (s != 0.0) {
                    res.col_idx_[pos] = col_idx_[a];
                    res.val_[pos] = s;
                    pos++;
                }
                a++;
                b++;
            }
        }
    }

    res.last_row_ = res.rows_;
    return res;
}

CSRMatrix CSRMatrix::mul_scalar(double alpha) const {
    CSRMatrix res(rows_, cols_);
    int n = nnz();

    res.row_ptr_ = row_ptr_;
    res.col_idx_ = col_idx_;
    res.val_.resize(n);

    for (int i = 0; i < n; i++) {
        res.val_[i] = val_[i] * alpha;
    }
    return res;
}

std::vector<double> CSRMatrix::mul_vector(const std::vector<double>& x) const {
    std::vector<double> res(rows_, 0.0);

    for (int r = 0; r < rows_; r++) {
        double sum = 0.0;
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];
        for (int k = start; k < end; k++) {
            sum += val_[k] * x[col_idx_[k]];
        }
        res[r] = sum;
    }
    return res;
}

CSRMatrix CSRMatrix::transpose() const {
    int n = nnz();
    CSRMatrix res(cols_, rows_);

    // Количество элементов в каждом столбце
    std::vector<int> col_count(cols_, 0);
    for (int r = 0; r < rows_; r++) {
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];
        for (int k = start; k < end; k++) {
            col_count[col_idx_[k]]++;
        }
    }

    // Создание новых значений строк
    res.row_ptr_[0] = 0;
    for (int c = 0; c < cols_; c++) {
        res.row_ptr_[c + 1] = res.row_ptr_[c] + col_count[c];
    }

    res.col_idx_.resize(n);
    res.val_.resize(n);

    // offset[c] - следующая свободная позиция в строке c транспонированной матрицы
    std::vector<int> offset = res.row_ptr_;

    // Последовательная запись строк в новую матрицу по столбцам предыдущей по порядку
    for (int r = 0; r < rows_; r++) {
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];

        for (int k = start; k < end; k++) {
            int c = col_idx_[k];

            int pos;
            pos = offset[c]++;
            res.col_idx_[pos] = r;
            res.val_[pos] = val_[k];
        }
    }

    res.last_row_ = res.rows_;
    return res;
}

CSRMatrix CSRMatrix::add(const CSRMatrix& other) const {
    CSRMatrix res(rows_, cols_);

    // Количество ненулевых элементов в новой матрице в каждой строке
    std::vector<int> row_nnz(rows_, 0);

    for (int r = 0; r < rows_; r++) {
        int a = row_ptr_[r], a_end = row_ptr_[r + 1];
        int b = other.row_ptr_[r], b_end = other.row_ptr_[r + 1];

        int count = 0;
        while (a < a_end || b < b_end) {
            if (a >= a_end || b >= b_end) {
                count++;
                a++;
                b++;
                continue;
            }
            if (a < a_end && col_idx_[a] < other.col_idx_[b]) {
                count++;
                a++;
            }
            else if (b < b_end && other.col_idx_[b] < col_idx_[a]) {
                count++;
                b++;
            }
            else {
                if (val_[a] + other.val_[b] != 0.0) count++;
                a++;
                b++;
            }
        }
        row_nnz[r] = count;
    }

    // Создание новых значений строк
    res.row_ptr_[0] = 0;
    for (int r = 0; r < rows_; r++) {
        res.row_ptr_[r + 1] = res.row_ptr_[r] + row_nnz[r];
    }

    int total = res.row_ptr_[rows_];
    res.col_idx_.resize(total);
    res.val_.resize(total);

    // Последовательная запись строк в новую матрицу
    for (int r = 0; r < rows_; r++) {
        int a = row_ptr_[r], a_end = row_ptr_[r + 1];
        int b = other.row_ptr_[r], b_end = other.row_ptr_[r + 1];

        int pos = res.row_ptr_[r];

        while (a < a_end || b < b_end) {
            if (a >= a_end) {
                res.col_idx_[pos] = other.col_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
                continue;
            }
            if (b >= b_end) {
                res.col_idx_[pos] = col_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
                continue;
            }
            if (a < a_end && col_idx_[a] < other.col_idx_[b]) {
                res.col_idx_[pos] = col_idx_[a];
                res.val_[pos] = val_[a];
                pos++;
                a++;
            }
            else if (b < b_end && other.col_idx_[b] < col_idx_[a]) {
                res.col_idx_[pos] = other.col_idx_[b];
                res.val_[pos] = other.val_[b];
                pos++;
                b++;
            }
            else {
                double s = val_[a] + other.val_[b];
                if (s != 0.0) {
                    res.col_idx_[pos] = col_idx_[a];
                    res.val_[pos] = s;
                    pos++;
                }
                a++;
                b++;
            }
        }
    }

    res.last_row_ = res.rows_;
    return res;
}

void CSRMatrix::print_matrix() const {
    for (int r = 0; r < rows_; r++) {
        int start = row_ptr_[r];
        int end = row_ptr_[r + 1];

        int k = start;
        for (int c = 0; c < cols_; c++) {
            double v = 0.0;
            if (k < end && col_idx_[k] == c) {
                v = val_[k];
                k++;
            }
            std::cout << v << " ";
        }
        std::cout << "\n";
    }
}

std::string CSRMatrix::repr() const {
    std::vector<double> dense(rows_ * cols_, 0.0);

    for (int r = 0; r < rows_; r++)
        for (int idx = row_ptr_[r]; idx < row_ptr_[r + 1]; ++idx)
            dense[r * cols_ + col_idx_[idx]] = val_[idx];

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

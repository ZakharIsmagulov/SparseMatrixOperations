#include "COOMatrix.h"
#include "DenseMatrix.h"
#include "CSRMatrix.h"
#include "CSCMatrix.h"
#include <iostream>
#include <omp.h> 
#include <string>
#include "VecRepr.h"

template <typename MatrixType>
void add_bench(const MatrixType& m1, const MatrixType& m2) {
	double start_p, end_p, start_s, end_s;
	start_p = omp_get_wtime();
	MatrixType res1 = m1.add_p(m2);
	end_p = omp_get_wtime();

	start_s = omp_get_wtime();
	MatrixType res2 = m1.add(m2);
	end_s = omp_get_wtime();

	std::cout << "Parallel:   " << (end_p - start_p) << " s\n";
	std::cout << "Sequential: " << (end_s - start_s) << " s\n";
}

template <typename MatrixType>
void transpose_bench(const MatrixType& m) {
	double start_p, end_p, start_s, end_s;
	start_p = omp_get_wtime();
	MatrixType res1 = m.transpose_p();
	end_p = omp_get_wtime();

	start_s = omp_get_wtime();
	MatrixType res2 = m.transpose();
	end_s = omp_get_wtime();

	std::cout << "Parallel:   " << (end_p - start_p) << " s\n";
	std::cout << "Sequential: " << (end_s - start_s) << " s\n";
}

template <typename MatrixType>
void mul_vector_bench(const MatrixType& m, const std::vector<double>& vector) {
	double start_p, end_p, start_s, end_s;
	start_p = omp_get_wtime();
	std::vector<double> res1 = m.mul_vector_p(vector);
	end_p = omp_get_wtime();

	start_s = omp_get_wtime();
	std::vector<double> res2 = m.mul_vector(vector);
	end_s = omp_get_wtime();

	std::cout << "Parallel:   " << (end_p - start_p) << " s\n";
	std::cout << "Sequential: " << (end_s - start_s) << " s\n";
}

template <typename MatrixType>
void mul_scalar_bench(const MatrixType& m, double alpha) {
	double start_p, end_p, start_s, end_s;
	start_p = omp_get_wtime();
	MatrixType res1 = m.mul_scalar_p(alpha);
	end_p = omp_get_wtime();

	start_s = omp_get_wtime();
	MatrixType res2 = m.mul_scalar(alpha);
	end_s = omp_get_wtime();

	std::cout << "Parallel:   " << (end_p - start_p) << " s\n";
	std::cout << "Sequential: " << (end_s - start_s) << " s\n";
}

void print_metrics(const std::string m5per_path, const std::string m25per_path, const std::string vec_path) {
	COOMatrix coo1(m5per_path);
	DenseMatrix dense1(coo1);
	CSRMatrix csr1(coo1);
	CSCMatrix csc1(coo1);

	COOMatrix coo2(m25per_path);
	DenseMatrix dense2(coo2);
	CSRMatrix csr2(coo2);
	CSCMatrix csc2(coo2);

	std::vector<double> vector = read_vector(vec_path);
	double alpha = 1354.564132;

	std::cout << "Multiply by alpha for 5% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	mul_scalar_bench(dense1, alpha);
	std::cout << "COO Matrix:\n";
	mul_scalar_bench(coo1, alpha);
	std::cout << "CSR Matrix:\n";
	mul_scalar_bench(csr1, alpha);
	std::cout << "CSC Matrix:\n";
	mul_scalar_bench(csc1, alpha);

	std::cout << "\n";

	std::cout << "Multiply by alpha for 25% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	mul_scalar_bench(dense2, alpha);
	std::cout << "COO Matrix:\n";
	mul_scalar_bench(coo2, alpha);
	std::cout << "CSR Matrix:\n";
	mul_scalar_bench(csr2, alpha);
	std::cout << "CSC Matrix:\n";
	mul_scalar_bench(csc2, alpha);

	std::cout << "\n\n";

	std::cout << "Multiply by vector for 5% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	mul_vector_bench(dense1, vector);
	std::cout << "COO Matrix:\n";
	mul_vector_bench(coo1, vector);
	std::cout << "CSR Matrix:\n";
	mul_vector_bench(csr1, vector);
	std::cout << "CSC Matrix:\n";
	mul_vector_bench(csc1, vector);

	std::cout << "\n";

	std::cout << "Multiply by vector for 25% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	mul_vector_bench(dense2, vector);
	std::cout << "COO Matrix:\n";
	mul_vector_bench(coo2, vector);
	std::cout << "CSR Matrix:\n";
	mul_vector_bench(csr2, vector);
	std::cout << "CSC Matrix:\n";
	mul_vector_bench(csc2, vector);

	std::cout << "\n\n";

	std::cout << "Transposing for 5% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	transpose_bench(dense1);
	std::cout << "COO Matrix:\n";
	transpose_bench(coo1);
	std::cout << "CSR Matrix:\n";
	transpose_bench(csr1);
	std::cout << "CSC Matrix:\n";
	transpose_bench(csc1);

	std::cout << "\n";

	std::cout << "Transposing for 25% matrix time:\n";
	std::cout << "Dense Matrix:\n";
	transpose_bench(dense2);
	std::cout << "COO Matrix:\n";
	transpose_bench(coo2);
	std::cout << "CSR Matrix:\n";
	transpose_bench(csr2);
	std::cout << "CSC Matrix:\n";
	transpose_bench(csc2);

	std::cout << "\n\n";
	
	std::cout << "Adding 2 matrices for 5% and 25% time:\n";
	std::cout << "Dense Matrix:\n";
	add_bench(dense1, dense2);
	std::cout << "COO Matrix:\n";
	add_bench(coo1, coo2);
	std::cout << "CSR Matrix:\n";
	add_bench(csr1, csr2);
	std::cout << "CSC Matrix:\n";
	add_bench(csc1, csc2);
}

int main() {

	print_metrics("matrices_txt/COO11.txt", "matrices_txt/COO12.txt", "matrices_txt/Vec1.txt");
	print_metrics("matrices_txt/COO21.txt", "matrices_txt/COO22.txt", "matrices_txt/Vec2.txt");
	print_metrics("matrices_txt/COO31.txt", "matrices_txt/COO32.txt", "matrices_txt/Vec3.txt");

	return 0;
}
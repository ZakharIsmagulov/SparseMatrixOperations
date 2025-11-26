#pragma once

#include <vector>
#include <string>

std::vector<double> read_vector(const std::string& filename);
std::string format2(double x);
std::string repr_vector(const std::vector<double>& v);
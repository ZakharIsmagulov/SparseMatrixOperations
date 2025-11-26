#include "VecRepr.h"

std::vector<double> read_vector(const std::string& filename)
{
    FILE* f = std::fopen(filename.c_str(), "r");

    int n;
    std::fscanf(f, "%d", &n);

    std::vector<double> vec(n);

    for (int i = 0; i < n; ++i) {
        std::fscanf(f, "%lf", &vec[i]);
    }

    std::fclose(f);
    return vec;
}

std::string format2(double x) {
    std::string s = std::to_string(x);
    size_t p = s.find('.');
    if (p == std::string::npos)
        return s + ".00";
    return s.substr(0, p + 3);
}

std::string repr_vector(const std::vector<double>& v) {
    std::string out;

    for (size_t i = 0; i < v.size(); ++i) {
        out += format2(v[i]);
        if (i + 1 < v.size())
            out += " ";
    }

    return out;
}

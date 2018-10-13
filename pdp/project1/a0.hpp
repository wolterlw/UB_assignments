// File: a0.hpp
// Volodymyr
// Liunda
// I AFFIRM THAT WHEN WORKING ON THIS ASSIGNMENT
// I WILL FOLLOW UB STUDENT CODE OF CONDUCT, AND
// I WILL NOT VIOLATE ACADEMIC INTEGRITY RULES

#ifndef A0_HPP
#define A0_HPP

#include <vector>
#include <iostream>

void cumsum(bool* in, long* out, long size){
	out[0] = in[0];
	for (int i=1; i < size; i++)
		out[i] = out[i-1] + in[i];
}
// IMPLEMENT ME!
template <typename T, typename Pred>
void omp_extract_if(const std::vector<T>& in, std::vector<T>& out, Pred pred) {

	int in_size = in.size();
	int i, out_size;
	
	bool* flags = new bool[in_size];
	long int* cums = new long int[in_size];

	#pragma omp parallel shared(flags, in), private(i)
	for (i=0; i < in_size;i++) flags[i] = pred(in[i]);

	cumsum(flags, cums, in_size);
	out_size = cums[in_size-1];
	
	out.resize(out_size);
	#pragma omp parallel shared(flags, in, out), private(i)
	for (i=0; i < in_size; i++){
		if (flags[i]) out[cums[i]-1] = in[i];
	}

	delete[] flags;
	delete[] cums;
} // omp_extract_if

#endif // A0_HPP

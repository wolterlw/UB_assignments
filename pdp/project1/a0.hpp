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

void omp_cumsum(long int* x, long int n){
    long int add;
    long int prev;
    int i;

    for (long int cur=2; cur<=n; cur*=2){
    	prev = cur / 2;

    	#pragma omp parallel for private(i), shared(n, cur, prev, x)
    	for (i = 0; i<n/cur; i++){
    		x[(i+1) * cur - 1] = x[(i+1) * cur - 1] + x[(2*i+1)*prev - 1];
		}

		//not yet parallel beucase of "add"
    	if (cur > 4){
    		for (i = prev + 3; i <= n; i+=2){
    			if ((i-3) % prev == 0) add = i-4;  //CAUTION Race Condition!!!!
    			if ((prev+2 < i % cur) and (i % cur < cur)) x[i-2] = x[add] + x[i-2];
    		}
    	}
    }

    #pragma omp parallel for private(i), shared(n,x)
    for (int i=2; i<n; i+=2) x[i] = x[i] + x[i-1];
}

// QUESTION can I have a parallel clause outside of the function
// and only have omp for inside?

// IMPLEMENT ME!
template <typename T, typename Pred>
void omp_extract_if(const std::vector<T>& in, std::vector<T>& out, Pred pred) {

	int in_size = in.size();
	int i, out_size;
	
	bool* flags = new bool[in_size];
	long int* cums = new long int[in_size];

	#pragma omp parallel for shared(flags, in), private(i)
	for (i=0; i < in_size;i++) flags[i] = pred(in[i]);

	#pragma omp parallel for shared(flags, cums), private(i)
	for (i=0; i < in_size; i++) cums[i] = flags[i];

	omp_cumsum(cums, in_size);
	out_size = cums[in_size-1];
	
	out.resize(out_size);
	#pragma omp parallel for shared(flags, in, out), private(i)
	for (i=0; i < in_size; i++){
		if (flags[i]) out[cums[i]-1] = in[i];
	}

	delete[] flags;
	delete[] cums;
} // omp_extract_if

#endif // A0_HPP

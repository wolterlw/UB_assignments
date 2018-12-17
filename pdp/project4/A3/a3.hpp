/*  Volodymyr
 *  Liunda
 *  vliunda
 */

#include <cmath>
// #include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
// #include <random>
// #include <chrono>
// #include <stdio.h>
// #include <fstream>

#ifndef A3_HPP
#define A3_HPP

#define SQRT2PI 2.50662827463100050241
#define BLOCK_SIZE 1024
#define SHARED_MEM_SIZE 4096

__device__ float gauss_kernel(float x){
    return exp(-x*x / 2) / SQRT2PI;
}

__global__
void kernel(int n, float h, float* x, float* y){
	__shared__ float x_buf[SHARED_MEM_SIZE];
	__shared__ float y_buf[SHARED_MEM_SIZE];
	float xr;

	int gidx  = blockIdx.x * blockDim.x + threadIdx.x;
	if (gidx < n){
 		xr = x[gidx];

 		for (int i = 0; i < gridDim.x; i++){
 			int j = blockDim.x*((blockIdx.x + i) % gridDim.x) + threadIdx.x;
 			x_buf[threadIdx.x] = x[j];
			__syncthreads();
			
 			for (int k = 0; k < blockDim.x; k++){
 				y_buf[threadIdx.x] += gauss_kernel((xr - x_buf[k]) / h);
 			}
 		}
	y_buf[threadIdx.x] /= (n*h);
	y[gidx] = y_buf[threadIdx.x];
	}
}

__host__ 
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
	float* d_x;
	float* d_y;

	int size = n * sizeof(float);
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);

	cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);

	int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	std::cout << numBlocks << std::endl ;
	kernel<<<numBlocks, BLOCK_SIZE>>>(n, h, d_x, d_y);
	cudaDeviceSynchronize();

	cudaMemcpy(y.data(), d_y, size, cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_y);
} // gaussian_kde

#endif // A3_HPP

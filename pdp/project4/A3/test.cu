#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <stdio.h>

#define SQRT2PI 2.50662827463100050241

__device__ float gauss_kernel(float x){
    return exp(-x*x / 2) / SQRT2PI;
}

// __global__ void apply_kernel(int n, float h, float* x, float* y){
// 	__shared__ float* y_buf;
// 	__shared__ float* x_buf;

// 	cudaMalloc(&y_buf, n * sizeof(float));
// 	cudaMalloc(&x_buf, n * sizeof(float));

// 	int gidx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (gidx < n){
// 		float xr = y[gidx];
// 		y_buf[threadIdx.x] = y[gidx];

// 		for (int i = 0; i < n; i+=blockDim.x){
// 			j = blockDim.x*(blockIdx.x + j) + threadIdx.x;
// 			x_buf[threadIdx.x] = x[j];

// 			for (int k = 0; k < blockDim.x; k++){
// 				y_buf[threadIdx.x] += gauss_kernel((xr - x_buf[k]) / h);
// 			}
// 		}
// 	}
	

// }

__global__
void print(int n, float h, float* x, float* y){
	extern __shared__ float  shared_mem[];
	float* y_buf = &shared_mem[0];
	float* x_buf = &shared_mem[blockDim.x];
	float xr;

	int gidx  = blockIdx.x * blockDim.x + threadIdx.x;
	if (gidx < n){
 		xr = x[gidx];
 		y_buf[threadIdx.x] = y[gidx];

 		for (int i = 0; i < n; i+=gridDim.x){
 			int j = blockDim.x*(blockIdx.x + i) + threadIdx.x;
 			x_buf[threadIdx.x] = x[j];
			__syncthreads();
 			for (int k = 0; k < blockDim.x; k++){
 				y_buf[threadIdx.x] += gauss_kernel((xr - x_buf[k]) / h);
 			}
 		}
 	}
	y_buf[threadIdx.x] /= (n*h);

	printf("%d: %f\t%f\n", gidx, y_buf[threadIdx.x], xr);
}

__host__ 
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y){
	float* d_x;
	float* d_y;

	int size = n * sizeof(float);
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);

	cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	std::cout << numBlocks << std::endl ;
	print<<<numBlocks, blockSize, 2 * blockSize*sizeof(float)>>>(n, h, d_x, d_y);
	
	cudaDeviceSynchronize();

	cudaMemcpy(d_y, y.data(), size, cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
}

int main(int argc, char const *argv[])
{

	int n = 512;
	float h = 0.1;

	std::vector<float> x(n);
    std::vector<float> y(n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<float> N(0.0, 1.0);
    std::generate(std::begin(x), std::end(x), std::bind(N, gen));

    // now running your awesome code from a3.hpp
    auto t0 = std::chrono::system_clock::now();

    gaussian_kde(n, h, x, y);

    auto t1 = std::chrono::system_clock::now();

    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout << "Tp: " << elapsed_par.count() << "s" << std::endl;
	return 0;
}

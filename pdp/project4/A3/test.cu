#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <stdio.h>
#include <fstream>

#define SQRT2PI 2.50662827463100050241
#define BLOCK_SIZE 512
#define SHARED_MEM_SIZE 2048

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
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y){
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
}

int main(int argc, char const *argv[])
{

	int n = 1000000;
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
    std::ofstream x_file;
    std::ofstream y_file;

    x_file.open("x.csv");
    y_file.open("y.csv");

    for (auto i: x) x_file << i << ",";
    for (auto i: y) y_file << i << ",";
    x_file.close();
    y_file.close();
	return 0;
}

#include <cmath>

#define SQRT2PI 2.50662827463100050241

__device__ float gauss_kernel(float x){
	return exp(-x*x / 2) / SQRT2PI
}

__global__ void apply_kernel(int n, float h, float* x, float* y){
	__shared__ float* y_buf;
	__shared__ float* x_buf;

	cudaMalloc(&y_buf, n * sizeof(float));
	cudaMalloc(&x_buf, n * sizeof(float));

	int gidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gidx < n){
		float xr = y[gidx];
		y_buf[threadIdx.x] = y[gidx];
		
		for (int i = 0; i < n; i+=blockDim.x){
			j = blockDim.x*(blockIdx.x + j) + threadIdx.x;
			x_buf[threadIdx.x] = x[j];

			for (int k = 0; k < blockDim.x; k++){
				y_buf[threadIdx.x] += gauss_kernel((xr - x_buf[k]) / h);
			}
		}
	}
	

}
__host__ 
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y){
	float* d_x;
	float* d_y;

	int size = n * sizeof(float)
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);

	cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	apply_kernel<<<numBlocks, blockSize>>>(n, h, d_x, d_y);
	
	cudaDeviceSynchronize();

	cudaMemcpy(d_y, y.data(), size, cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
}

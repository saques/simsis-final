#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "Vector.h"

#define HANDLE_CUDA_ERROR(ERROR) if ((ERROR) != cudaSuccess) { \
		fprintf(stderr, "Cuda memory error"); \
		return 1; \
		} \

int main()
{
	// Init Cuda
	cudaError_t status;
	status = cudaSetDevice(0);
	Vec3 vec1cpu[100], vec2cpu[100], vec3cpu[100];
	// Init arrays with whatever
	for (int i = 0; i < 100; i++) vec1cpu[i] = {2.0f * i, 7.0f * i, 13.0f * i };
	for (int i = 0; i < 100; i++) vec2cpu[i] = {3.0f * i, 11.0f * i, 17.0f * i };
	
	Vec3 *vec1 = {0}, *vec2 = {0}, *vec3 = {0};
	size_t vecArrSize = 100 * sizeof(Vec3);

	// Alloc 3, 100 Vec3 arrays
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec1, vecArrSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec2, vecArrSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec3, vecArrSize));
	
	// Memcpy vec1, vec2
	HANDLE_CUDA_ERROR(cudaMemcpy(vec1, vec1cpu, vecArrSize, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(vec2, vec2cpu, vecArrSize, cudaMemcpyHostToDevice));

	// Add vectors in parallel.
	sum<<<1, 100>>>(vec1, vec2, vec3);

	// Sync device
	HANDLE_CUDA_ERROR(cudaGetLastError());
	HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
	HANDLE_CUDA_ERROR(cudaMemcpy(vec3cpu, vec3, vecArrSize, cudaMemcpyDeviceToHost));
	// Copy result to CPU

	for (int i = 0; i < 100; i++) {
		printf("Vec3 {%f, %f, %f} \n", vec3cpu[i].x, vec3cpu[i].y, vec3cpu[i].z);
	}

	cudaFree(vec1);
	cudaFree(vec2);
	cudaFree(vec3);
	return 0;
}
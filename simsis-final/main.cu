#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include "cuda_runtime.h"

#include "classes/Particle.hpp"
#include "classes/Vector.hpp"
#include "classes/Grid.hpp"
#include "kernels/test.cu"

#define HANDLE_CUDA_ERROR(ERROR) if ((ERROR) != cudaSuccess) { \
		fprintf(stderr, "Cuda memory error"); \
		return 1; \
		} \


__host__  void writeToFile(Grid<Particle> * grid, std::ofstream &file) {

	int size = grid->getRows() * grid->getCols();

	file << size << std::endl;
	file << std::endl;

	for (int row = 0; row < grid->getRows(); row++)
		for (int col = 0; col < grid->getCols(); col++) {
			Vec3 vec = grid->get(row, col).position;
			file << vec.x << " " << vec.y << " " << vec.z << std::endl;
		}
}

//A test to show how to work with classes and CUDA
//See Grid.hpp and test.cu
int deviceCompatibleClassExample() {
	Grid<int> * g = new Grid<int>(1, 1);
	Grid<int> * g_device = Grid<int>::gridcpy(g, Grid<int>::UPLOAD);

	change << <1, 1 >> > (g_device, 0, 0, 4);

	Grid<int> * d = Grid<int>::gridcpy(g_device, Grid<int>::DOWNLOAD);

	std::cout << d->get(0, 0) << std::endl;

	return 0;
}

int vectorsExample() {
	Vec3 vec1cpu[100], vec2cpu[100], vec3cpu[100];

	// Init arrays with whatever
	for (int i = 0; i < 100; i++) vec1cpu[i] = { 2.0f * i, 7.0f * i, 13.0f * i };
	for (int i = 0; i < 100; i++) vec2cpu[i] = { 3.0f * i, 11.0f * i, 17.0f * i };

	Vec3 *vec1 = { 0 }, *vec2 = { 0 }, *vec3 = { 0 };
	size_t vecArrSize = 100 * sizeof(Vec3);

	// Alloc 3, 100 Vec3 arrays
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec1, vecArrSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec2, vecArrSize));
	HANDLE_CUDA_ERROR(cudaMalloc((void **)&vec3, vecArrSize));

	// Memcpy vec1, vec2
	HANDLE_CUDA_ERROR(cudaMemcpy(vec1, vec1cpu, vecArrSize, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(vec2, vec2cpu, vecArrSize, cudaMemcpyHostToDevice));

	// Add vectors in parallel.
	sum << <1, 100 >> > (vec1, vec2, vec3);

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

#define DUMP_FILE "particles.dump"

int main(){
	cudaError_t status;

	int rows = 100, cols = 100;
	int ticks = 100;

	Grid<Particle> * g = new Grid<Particle>(rows, cols);
	Grid<Particle> * g_device = Grid<Particle>::gridcpy(g, Grid<Particle>::UPLOAD);

	dim3 dimBlock = dim3(10, 10);
	int yBlocks = cols / dimBlock.y + ((cols%dimBlock.y) == 0 ? 0 : 1);
	int xBlocks = rows / dimBlock.x + ((rows%dimBlock.x) == 0 ? 0 : 1);
	dim3 dimGrid = dim3(xBlocks, yBlocks);


	initializePositions <<<dimGrid, dimBlock >>> (g_device, 1);
	Grid<Particle> * d;
	d = Grid<Particle>::gridcpy(g_device, Grid<Particle>::DOWNLOAD);
	
	std::ofstream file(DUMP_FILE);
	for (int i = 0; i < ticks; i++) {
		//Try moveDownwardsCool!
		moveDownwards << <dimGrid, dimBlock >> > (g_device, 1);
		d = Grid<Particle>::gridcpy(g_device, Grid<Particle>::DOWNLOAD);

		writeToFile(d, file);
	}
	file.close();
	
	status = cudaDeviceReset();
	return 0;
}
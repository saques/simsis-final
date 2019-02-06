#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <Windows.h>
#include "cuda_runtime.h"

#include "classes/Particle.hpp"
#include "classes/Vector.hpp"
#include "classes/Grid.hpp"
#include "kernels/test.cu"
#include "kernels/grid_kernels.cu"

#define HANDLE_CUDA_ERROR(ERROR) if ((ERROR) != cudaSuccess) { \
		fprintf(stderr, "Cuda memory error"); \
		return 1; \
		} \

#define FRAME_RATE 60

__host__  void writeToFile(Grid<Particle> * grid, std::ofstream &file, Particle * big, int size) {
	int s = grid->getRows() * grid->getCols() + size;
	char * buf = new char[s * 50 * 4];
	char * bufp = (char *) buf;

	int len = sprintf(bufp, "%d \n\n", s);
	bufp += len;

	for (int i = 0; i < size; i++) {
		Particle p = big[i];
		Vec3 vec = p.position;
		int len = sprintf(bufp, "%f %f %f %f\n", vec.x, vec.y, vec.z, p.radius);
		bufp += len;
	}

	for (int row = 0; row < grid->getRows(); row++)
		for (int col = 0; col < grid->getCols(); col++) {
			Particle p = grid->get(row, col);
			Vec3 vec = p.position;
			int len = sprintf(bufp, "%f %f %f %f\n", vec.x, vec.y, vec.z, p.radius);
			bufp += len;
		}
	file.write(buf, bufp - buf);
	delete[] buf;
}

__host__ void computeBigMassForces(Particle * p_host, int p_size, Grid<Particle> * g_device, float g, float separation, float kn, float kt, float natural, float kbig) {

	Particle * p_device;
	cudaMalloc((void **)&(p_device), sizeof(Particle)*p_size);
	cudaMemcpy(p_device, p_host, sizeof(Particle)*p_size, cudaMemcpyHostToDevice);

	//Reset forces in particles resetBigParticles(Particle * particles, int size, float g)
	resetBigParticles<<<1, p_size>>>(p_device, p_size, g);

	interactBigParticles << <1, p_size >> > (p_device, p_size, natural, kbig);

	//Interact grid and particle
	int x_start = (int)((p_host->position.x - p_host->radius) / separation);
	int y_start = (int)((p_host->position.y - p_host->radius) / separation);

	int x_end = x_start + (int)(2 * p_host->radius / separation);
	int y_end = y_start + (int)(2 * p_host->radius / separation);


	dim3 dimBlock = dim3(10, 10);
	int yBlocks = (y_end - y_start) / dimBlock.y + (((y_end - y_start) % dimBlock.y) == 0 ? 0 : 1);
	int xBlocks = (x_end - x_start) / dimBlock.x + (((x_end - x_start) % dimBlock.x) == 0 ? 0 : 1);
	dim3 dimGrid = dim3(xBlocks, yBlocks);

	//__global__ void interactGridAndParticle(Grid<Particle> * grid, Particle * big, int start_x, int start_y, int end_x, int end_y, float kn) {

	for(int i = 0; i < p_size; i++)
		interactGridAndParticle << <dimGrid, dimBlock >> > (g_device, p_device, i, x_start, y_start, x_end, y_end, kn, kt);

	cudaMemcpy(p_host, p_device, sizeof(Particle)*p_size, cudaMemcpyDeviceToHost);
	cudaFree(p_device);
}


__host__ void updateEulerBigMass(Particle * p_host, int p_size, float delta_t) {

	Particle * p_device;
	cudaMalloc((void **)&(p_device), sizeof(Particle)*p_size);
	cudaMemcpy(p_device, p_host, sizeof(Particle)*p_size, cudaMemcpyHostToDevice);

	updateEulerBigParticles << <1, p_size >> > (p_device, p_size, delta_t);

	cudaMemcpy(p_host, p_device, sizeof(Particle)*p_size, cudaMemcpyDeviceToHost);
	cudaFree(p_device);
}





#define DUMP_FOLDER "dump"
#define THREAD_COUNT 8
int main(){
	
	cudaError_t status;

	float simulation_t = 5;
	float delta_t = 0.0001f;
	int rows = 75, cols = 75;
	int frame_rate = 60;
	float separation = 0.05, mass = 0.005, radius = 0.01, g_earth = 9.81, k = 3000;
	int skip_x = 1, skip_y = 1;

	float big_mass = 3, big_radius = 0.15, kn = 1E5, kt = 1E3, separation_big = big_radius, kbig = 1E8;
	Vec3 big_init = { rows/2*separation, rows/2*separation, 3 };

	int ticks = simulation_t/delta_t;
	int dump_each = (int) ((1.0 / frame_rate) / delta_t);

	Grid<Particle> * g = new Grid<Particle>(rows, cols);
	Grid<Particle> * g_device = Grid<Particle>::gridcpy(g, Grid<Particle>::UPLOAD);

	int big_size = 2;
	Particle * big = newParticles(big_init, big_mass, big_radius, big_size, separation_big);

	dim3 dimBlock = dim3(10, 10);
	int yBlocks = cols / dimBlock.y + ((cols%dimBlock.y) == 0 ? 0 : 1);
	int xBlocks = rows / dimBlock.x + ((rows%dimBlock.x) == 0 ? 0 : 1);
	dim3 dimGrid = dim3(xBlocks, yBlocks);

	std::chrono::high_resolution_clock clock;
	
	initializePositions <<<dimGrid, dimBlock >>> (g_device, separation, mass, radius);
	Grid<Particle> * d;
	
	auto pre = clock.now();

	std::queue<Grid<Particle>*> q;
	int count = 0;
	std::mutex m;

	CreateDirectory(DUMP_FOLDER, nullptr);
	auto writer = [&q, ticks, dump_each, &count, &m, big, big_size] {
		int total_count = ticks / dump_each;
		while (count < total_count) {
			Grid<Particle>* el = nullptr;
			int c = 0;
			m.lock();
			if (!q.empty()) {
				el = q.front();
				q.pop();
				c = count;
				count++;
			}
			m.unlock();
			if (el != nullptr) {
				std::ofstream file(std::string(DUMP_FOLDER) + "/" + std::to_string(c) + ".dump");
				writeToFile(el, file, big, big_size);
				delete el;
				file.close();
			}
			else {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
		}
	};

	std::thread t[THREAD_COUNT];
	for (int i = 0; i < THREAD_COUNT; i++) t[i] = std::thread(writer);

	for (int i = 0; i < ticks; i++) {
		//Try moveDownwardsCool!
		/*reset << <dimGrid, dimBlock >> > (g_device, g_earth, skip_x, skip_y);
		gridElasticForce << <dimGrid, dimBlock >> > (g_device, k, separation, skip_x, skip_y);
		updateEuler << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);*/
		
		verletPositions << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);
		reset << <dimGrid, dimBlock >> > (g_device, g_earth, skip_x, skip_y);
		gridElasticForce << <dimGrid, dimBlock >> > (g_device, k, separation, skip_x, skip_y);
		computeBigMassForces(big, big_size, g_device, g_earth, separation, kn, kt, separation_big, kbig);
		verletVelocities << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);
		updateEulerBigMass(big, big_size, delta_t);

		if (i % dump_each == 0) {
			// Dump to file
			d = Grid<Particle>::gridcpy(g_device, Grid<Particle>::DOWNLOAD);
			m.lock();
			q.push(d);
			m.unlock();
		}
	}
	printf("Waiting for disk operations...");
	auto waiting = clock.now();


	for (int i = 0; i < THREAD_COUNT; i++) t[i].join();

	auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - waiting);
	printf("Wasted %f seconds writing\n", waited / 1000.0);
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - pre);
	printf("Millis elapsed: %d\n", elapsed.count());

	
	status = cudaDeviceReset();
	return 0;
}
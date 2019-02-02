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

__host__  void writeToFile(Grid<Particle> * grid, std::ofstream &file) {
	int s = grid->getRows() * grid->getCols();
	char * buf = new char[s * 50 * 3];
	char * bufp = (char *) buf;
	
	int len = sprintf(bufp, "%d \n\n", s);
	bufp += len;
	for (int row = 0; row < grid->getRows(); row++)
		for (int col = 0; col < grid->getCols(); col++) {
			Vec3 vec = grid->get(row, col).position;
			int len = sprintf(bufp, "%f %f %f\n", vec.x, vec.y, vec.z);
			bufp += len;
		}
	file.write(buf, bufp - buf);
	delete[] buf;
}

#define DUMP_FOLDER "dump"
#define THREAD_COUNT 8
int main(){
	
	cudaError_t status;

	float simulation_t = 10;
	float delta_t = 0.001f;
	int rows = 100, cols = 100;
	int frame_rate = 60;
	float separation = 1, mass = 0.1, radius = 0.5, g_earth = 9.81, k = 1000;
	int skip_x = 10, skip_y = 10;

	int ticks = simulation_t/delta_t;
	int dump_each = (int) ((1.0 / frame_rate) / delta_t);

	Grid<Particle> * g = new Grid<Particle>(rows, cols);
	Grid<Particle> * g_device = Grid<Particle>::gridcpy(g, Grid<Particle>::UPLOAD);

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
	auto writer = [&q, ticks, dump_each, &count, &m] {
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
				writeToFile(el, file);
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
		verletVelocities << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);

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
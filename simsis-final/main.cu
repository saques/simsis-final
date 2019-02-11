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
#include "classes/Stats.hpp"

#define HANDLE_CUDA_ERROR(ERROR) if ((ERROR) != cudaSuccess) { \
		fprintf(stderr, "Cuda memory error\n"); \
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

__host__  void writeStats(Stats * stats, int size, std::string path) {
	std::ofstream file;
	file.open(path);

	for (int i = 0; i < size; i++) {
		Stats s = stats[i];
		file << s.COM_height << " " << s.deformation << " " << s.big_energy << " " << s.grid_energy << std::endl;
	}

	file.close();
}

__host__ void computeBigMassForces(Particle * p_host, int p_size, Grid<Particle> * g_device, float g, float separation, float kn, float kt, float natural, float kbig, float bbig) {

	Particle * p_device;
	cudaMalloc((void **)&(p_device), sizeof(Particle)*p_size);
	cudaMemcpy(p_device, p_host, sizeof(Particle)*p_size, cudaMemcpyHostToDevice);

	//Reset forces in particles resetBigParticles(Particle * particles, int size, float g)
	resetBigParticles<<<1, p_size>>>(p_device, p_size, g);

	interactBigParticles << <1, p_size >> > (p_device, p_size, natural, kbig, bbig);

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


__host__ void updateEulerBigMass(Particle * p_host, int p_size, float delta_t, bool dump, Stats * stats, int tick, float big_mass_separation, float g, int dump_each) {

	Particle * p_device;
	cudaMalloc((void **)&(p_device), sizeof(Particle)*p_size);
	cudaMemcpy(p_device, p_host, sizeof(Particle)*p_size, cudaMemcpyHostToDevice);

	updateEulerBigParticles << <1, p_size >> > (p_device, p_size, delta_t);

	cudaMemcpy(p_host, p_device, sizeof(Particle)*p_size, cudaMemcpyDeviceToHost);
	cudaFree(p_device);

	if (dump) {
		Stats stat = stats[tick/dump_each];

		//float COM_height, deformation, big_energy;
		stat.resetBig_energy();
		stat.setCOM_height(p_host, p_size);
		stat.setDeformation(p_host, p_size, big_mass_separation);
		stat.addBig_energy(p_host, p_size, g);

		stats[tick/dump_each] = stat;
	}
}


const char * GetOption(char * argv[], int argc, const char * option, const char * default) {
	const char * value = nullptr;
	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], option) == 0) {
			value = argv[i + 1];
			break;
		}
	}
	if (value == nullptr) {
		value = default;
	}
	printf("%-20s=\t%s\n", option, value);
	return value;
}


#define DUMP_FOLDER "dump"
#define THREAD_COUNT 8
 int main(int argc, char * argv[]){
	
	cudaError_t status;
	const char * method = GetOption(argv, argc, "--method", "verlet");
	float simulation_t = atof(GetOption(argv, argc, "--simulation-time", "5"));
	float delta_t = atof(GetOption(argv, argc, "--delta-time", "0.00025f"));
	int rows = atoi(GetOption(argv, argc, "--rows", "75"));
	int cols = atoi(GetOption(argv, argc, "--cols", "75"));
	int frame_rate = atoi(GetOption(argv, argc, "--frame-rate", "60"));
	float separation = atof(GetOption(argv, argc, "--separation", "0.05"));
	float mass = atof(GetOption(argv, argc, "--mass", "0.005"));
	float radius = atof(GetOption(argv, argc, "--radius", "0.01")); 
	float g_earth = atof(GetOption(argv, argc, "--gravity", "9.81")); 
	float k = atof(GetOption(argv, argc, "-k", "1.5E3"));
	float b = atof(GetOption(argv, argc, "--b-scale", "1")) * sqrtf(mass * k);	// Crit. Amort. : b = 2 * sqrt(mass * k)
	int skip_x = atoi(GetOption(argv, argc, "--skip-x", "1"));
	int skip_y = atoi(GetOption(argv, argc, "--skip-y", "1")); 

	float big_mass = atof(GetOption(argv, argc, "--big-mass", "5"));
	float big_radius = atof(GetOption(argv, argc, "--big-radius", "0.15"));
	float kn = atof(GetOption(argv, argc, "--kn", "1E5"));
	float kt = atof(GetOption(argv, argc, "--kt", "1E3"));
	float separation_big = atof(GetOption(argv, argc, "--sbig", std::to_string(big_radius).c_str()));
	float kbig = atof(GetOption(argv, argc, "--kbig", std::to_string(1E7).c_str()));
	float bbig = atof(GetOption(argv, argc, "--bbig", std::to_string(0).c_str()));
	float zinit = atof(GetOption(argv, argc, "--zinit", std::to_string(3).c_str()));
	int big_size = atoi(GetOption(argv, argc, "--big-size", "2"));
	bool sloped = atoi(GetOption(argv, argc, "--sloped", "0")) ? true : false;

	
	int ticks = simulation_t/delta_t;
	int dump_each = (int) ((1.0 / frame_rate) / delta_t);
	Stats * stats = (Stats *) malloc((ticks/dump_each) * sizeof(Stats));

	Grid<Particle> * g = new Grid<Particle>(rows, cols);
	Grid<Particle> * g_device = Grid<Particle>::gridcpy(g, Grid<Particle>::UPLOAD);

	Particle * big = newParticles(big_mass, big_radius, big_size, separation_big, rows, cols, big_size, zinit, separation, sloped);

	dim3 dimBlock = dim3(16, 16);
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
	int met;
	if (strcmp(method, "euler") == 0) {
		met = 1;
	}
	else if (strcmp(method, "verlet") == 0) {
		met = 2;
	}
	else if (strcmp(method, "rk4") == 0) {
		met = 3;
	}
	else {
		printf("Error: method % is not supported.\n", method);
		exit(1);
	}
	for (int i = 0; i < ticks; i++) {

		bool dump = i % dump_each == 0;

		switch (met) {
		case 1: {
			///-----------EULER------------
			reset << <dimGrid, dimBlock >> > (g_device, g_earth, skip_x, skip_y);
			gridElasticForce << <dimGrid, dimBlock >> > (g_device, k, b, separation, skip_x, skip_y);
			updateEuler << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);
			break;
		}
		case 2: {
			///-----------VERLET-----------
			verletPositions << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);
			reset << <dimGrid, dimBlock >> > (g_device, g_earth, skip_x, skip_y);
			gridElasticForce << <dimGrid, dimBlock >> > (g_device, k, b, separation, skip_x, skip_y);
			computeBigMassForces(big, big_size, g_device, g_earth, separation, kn, kt, separation_big, kbig, bbig);
			updateEulerBigMass(big, big_size, delta_t, dump, stats, i, separation_big, g_earth, dump_each);
			verletVelocities << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y);
			break;
		}
		case 3: {
			///------------RK4-------------
			computeBigMassForces(big, big_size, g_device, g_earth, separation, kn, kt, separation_big, kbig, bbig);
			updateEulerBigMass(big, big_size, delta_t, dump, stats, i, separation_big, g_earth, dump_each);
			initialVelAccel << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y, k, b, separation, mass, g_earth);
			rk4 << <dimGrid, dimBlock >> > (g_device, delta_t, skip_x, skip_y, k, b, separation, mass, g_earth);
			break;
		}
		}
		if (dump) {
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


	printf("Writing stats...");
	auto waitingStats = clock.now();
	writeStats(stats, (ticks/dump_each), std::string(DUMP_FOLDER) + "/stats.txt");
	auto waitedStats = std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - waitingStats);
	printf("Wasted %f seconds writing stats\n", waitedStats / 1000.0);

	
	status = cudaDeviceReset();
	return 0;
}
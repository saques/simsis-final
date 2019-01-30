#pragma once
#include "cuda_runtime.h"
#include "../classes/Grid.hpp"

//The following kernels are just a demo
//for working with classes in CUDA/C++
__global__ void test(int * i) {

	Grid<int> * g = new Grid<int>(1, 1);
	g->set(0, 0, 1);
	*i = g->get(0, 0);
	delete g;

}
__global__ void change(Grid<int> * g, int row, int col, int v) {
	g->set(row, col, v);
}


//Initialize positions of all particles in the grid
__global__ void initializePositions(Grid<Particle> * grid, int distance) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < grid->getRows() && y < grid->getCols()) {
		Particle p = grid->get(x, y);
		p.position.z = 0;
		p.position.x = x * distance;
		p.position.y = y * distance;
		grid->set(x, y, p);
	}

}

//A Linear Motion example 
__global__ void moveDownwards(Grid<Particle> * grid, int z_diff) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < grid->getRows() && y < grid->getCols()) {
		Particle p = grid->get(x, y);
		p.position.z -= z_diff;
		grid->set(x, y, p);
	}

}

//A Cool example (tm)
__global__ void moveDownwardsCool(Grid<Particle> * grid) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < grid->getRows() && y < grid->getCols()) {
		Particle p = grid->get(x, y);
		p.position.z -= sqrtf(powf(x,2)+powf(y,2));
		grid->set(x, y, p);
	}

}

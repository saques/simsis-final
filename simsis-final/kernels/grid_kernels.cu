#pragma once
#include <cstdio>
#include "cuda_runtime.h"
#include "../classes/Grid.hpp"
#include "../classes/Vector.hpp"

//Initialize positions of all particles in the grid
__global__ void initializePositions(Grid<Particle> * grid, float separation, float mass, float radius) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < grid->getRows() && y < grid->getCols()) {
		Particle p = grid->get(x, y);
		p.position = { x*separation, y*separation, 0 };
		p.velocity = { 0, 0, 0 };
		p.force = { 0, 0, 0 };
		p.mass = mass;

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
		p.position.z -= sqrtf(powf(x, 2) + powf(y, 2));
		grid->set(x, y, p);
	}

}


//Compute elastic force
__global__ void gridElasticForce(Grid<Particle> * grid, float k, float natural) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//Borders are fixed
	if (x < grid->getRows() - 1 && y < grid->getCols() - 1 && x != 0 && y != 0 ) {

		//Potential optimization: save this to shared block-level memory and read
		//from this memory (100x faster than global GPU memory) the particles
		//in the following loop
		Particle p = grid->get(x, y);

		for (int i = x - 1; i <= x + 1; i++) {
			for (int j = y - 1; j <= y + 1; j++) {


				if (i != x && j != y) {
					Particle o = grid->get(i, j);
					Vec3 relative;

					float dst = distance(&p.position, &o.position);
					rel(&p.position, &o.position, &relative);
					float elastic_force =  - k * (dst - natural);

					scl(&relative, elastic_force);
					sumf(&relative, &p.force);
				}


			}
		}

		grid->set(x, y, p);
	}

}

//Reset forces vector, but compute gravitational force
//Pass grid_dst as grid
__global__ void reset(Grid<Particle> * grid, float g) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//Borders are fixed
	if (x < grid->getRows() - 1 && y < grid->getCols() - 1 && x != 0 && y != 0) {

		Particle p = grid->get(x, y);
		p.force = { 0 ,0 , (-g) * p.mass };
		grid->set(x, y, p);

	}

}

//Pass grid_dst as grid
__global__ void updateEuler(Grid<Particle> * grid, float delta_t) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//Borders are fixed
	if (x < grid->getRows() - 1 && y < grid->getCols() - 1 && x != 0 && y != 0) {

		Particle p = grid->get(x, y);

		Vec3 delta_v = { p.force.x / p.mass, p.force.y / p.mass, p.force.z / p.mass };
		scl(&delta_v, delta_t);
		sumf(&delta_v, &p.velocity);

		Vec3 delta_x = { p.velocity.x, p.velocity.y, p.velocity.z };
		scl(&delta_x, delta_t);
		sumf(&delta_x, &p.position);

		grid->set(x, y, p);

	}

}

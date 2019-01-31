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

__device__ bool checkGridBounds(int row, int col, int m_row, int m_col, int skip_row, int skip_col) {
	if ((row == 0 || row == m_row) && (col == 0 || col == m_col))	
		return false;											
	if ((row == 0 || row == m_row) && col % skip_col == 0)
		return false;											
	if ((col == 0 || col == m_col) && row % skip_row == 0)		
		return false;
	return true;
}
	

//Compute elastic force
__global__ void gridElasticForce(Grid<Particle> * grid, float k, float natural, int skip_x, int skip_y) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;
	
	//Potential optimization: save this to shared block-level memory and read
	//from this memory (100x faster than global GPU memory) the particles
	//in the following loop
	Particle p = grid->get(x, y);

	for (int i = max(0, x - 1); i <= min(x+1, m_row); i++) {
		for (int j = max(0,y - 1); j <= min(y+1, m_col); j++) {
			if (!(i == x && j == y)) {
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

//Reset forces vector, but compute gravitational force
//Pass grid_dst as grid
__global__ void reset(Grid<Particle> * grid, float g, int skip_x, int skip_y) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;
	Particle p = grid->get(x, y);
	p.force = { 0 ,0 , (-g) * p.mass };
	grid->set(x, y, p);
}

//Pass grid_dst as grid
__global__ void updateEuler(Grid<Particle> * grid, float delta_t, int skip_x, int skip_y) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;

	Particle p = grid->get(x, y);

	Vec3 delta_v = { p.force.x / p.mass, p.force.y / p.mass, p.force.z / p.mass };
	scl(&delta_v, delta_t);
	sumf(&delta_v, &p.velocity);

	Vec3 delta_x = { p.velocity.x, p.velocity.y, p.velocity.z };
	scl(&delta_x, delta_t);
	sumf(&delta_x, &p.position);

	grid->set(x, y, p);
}

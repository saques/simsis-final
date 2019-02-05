#pragma once
#include <cstdio>
#include "cuda_runtime.h"
#include "../classes/Grid.hpp"
#include "../classes/Vector.hpp"
#include "../classes/Particle.hpp"

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
		p.acceleration = { 0, 0, 0 };
		p.radius = radius;
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
				Particle * o = grid->getRef(i, j);
				Vec3 relative;

				float dst = distance(&p.position, &o->position);
				rel(&p.position, &o->position, &relative, dst);
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

__global__ void verletPositions(Grid<Particle> * grid, float delta_t, int skip_x, int skip_y) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;

	Particle p = grid->get(x, y);
	Vec3 nextPos = p.position;
	Vec3 tmpVel = p.velocity;
	Vec3 tmpAccl = p.acceleration;
	scl(&tmpVel, delta_t);
	scl(&tmpAccl, 0.5f * delta_t * delta_t);
	sumf(&tmpAccl, &tmpVel);
	sumf(&tmpVel, &nextPos);

	p.position = nextPos;
	grid->set(x, y, p);
}

__global__ void verletVelocities(Grid<Particle> * grid, float delta_t, int skip_x, int skip_y) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;

	Particle p = grid->get(x, y);
	Vec3 currAccl = p.acceleration, nextAccl = p.force;
	{
		scl(&nextAccl, 1.0f / p.mass);
	}
	Vec3 nextVel = p.velocity;
	{
		// nextVel = currVel + 0.5 * (currAccl + nextAccl) * deltaT;
		Vec3 sumAccl = currAccl;
		sumf(&nextAccl, &sumAccl);
		scl(&sumAccl, 0.5f * delta_t);
		sumf(&sumAccl, &nextVel);
	}
	p.velocity = nextVel;
	p.acceleration = nextAccl;

	grid->set(x, y, p);
}

__global__ void interactGridAndParticle(Grid<Particle> * grid, Particle * big, int start_x, int start_y, int end_x, int end_y, float kn, float kt) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < 0 || y < 0 || x + start_x >= grid->getRows() || y + start_y >= grid->getCols())
		return;

	Particle p = grid->get(x + start_x, y + start_y);

	Vec3 p_pos = p.position;
	Vec3 o_pos = big->position;

	float dist = distance(&p_pos, &o_pos);

	if (dist >= big->radius + p.radius)
		return;

	float exn = (o_pos.x - p_pos.x) / dist;
	float eyn = (o_pos.y - p_pos.y) / dist;
	float ezn = (o_pos.z - p_pos.z) / dist;

	Vec3 normalForce = { exn, eyn, ezn };
	Vec3 tangentForce = { -ezn, -eyn, exn };

	float overlap = p.radius + big->radius - dist;

	float normalForceMag = (-1) * kn * overlap;

	Vec3 p_vel = p.velocity;
	Vec3 o_vel = big->velocity;
	Vec3 vel_rel;
	difff(&p_vel, &o_vel, &vel_rel);

	float tanForceMag = -kt * overlap * dotf(&vel_rel, &tangentForce);
	scl(&tangentForce, tanForceMag);
	
	//Add forces to particle in grid
	scl(&normalForce, normalForceMag);
	sumf(&normalForce, &p.force);
	sumf(&tangentForce, &p.force);

	//Add opposing forces in big particle
	scl(&normalForce, -1);
	scl(&tangentForce, -1);
	sumf(&normalForce, &(big->force));
	sumf(&tangentForce, &(big->force));

	grid->set(x + start_x, y + start_y, p);

}

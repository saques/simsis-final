#pragma once
#include <cstdio>
#include "cuda_runtime.h"
#include "../classes/Grid.hpp"
#include "../classes/Vector.hpp"
#include "../classes/Particle.hpp"

#define SQRT_2 sqrtf(2)

__device__ void applyElasticForce(Particle * p, Particle * o, float k, float b, float natural) {

	Vec3 relative;

	float dst = distance(&p->position, &o->position);
	rel(&p->position, &o->position, &relative, dst);
	float elastic_force = -k * (dst - natural);

	scl(&relative, elastic_force);

	Vec3 damp = { 0 };
	rel(&p->velocity, &o->velocity, &damp, 1);
	scl(&damp, -b);
	sumf(&damp, &relative);
	sumf(&relative, &p->force);
}

//Initialize positions of all particles in the grid
__global__ void initializePositions(Grid<Particle> * grid, float separation, float mass, float radius) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (x < grid->getRows() && y < grid->getCols()) {
		Particle p = grid->get(x, y);
		p.position = { x*separation, y*separation, 0 };
		p.velocity = { 0, 0, 0 };
		p.mass = mass;
		p.force = { 0, 0, -9.81f*p.mass };
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

__device__ bool isDiag(int x, int y, int i, int j) {
	return !(x == i || y == j);
}

__device__ bool checkGridBounds(int row, int col, int m_row, int m_col, int skip_row, int skip_col) {
	if (row > m_row || col > m_col)
		return false;
	if ((row == 0 || row == m_row) && (col == 0 || col == m_col))	
		return false;											
	if ((row == 0 || row == m_row) && col % skip_col == 0)
		return false;											
	if ((col == 0 || col == m_col) && row % skip_row == 0)		
		return false;
	return true;
}

//Compute elastic force
__global__ void gridElasticForce(Grid<Particle> * grid, float k, float b, float natural, int skip_x, int skip_y) {
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

	float diag_natural = natural * SQRT_2;
	
	for (int i = max(0, x - 1); i <= min(x+1, m_row); i++) {
		for (int j = max(0,y - 1); j <= min(y+1, m_col); j++) {
			if (!(i == x && j == y)) {
				Particle o = grid->get(i, j);
				applyElasticForce(&p, &o, k, b, isDiag(x,y,i,j) ? diag_natural : natural);
			}
		}
	}
	__syncthreads();
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

struct VelAccel {
	Vec3 vel, accel;
};

__device__ VelAccel calcVel(Grid<Particle> * grid, const VelAccel &pair, int x, int y, float delta_t, float k, float b, float natural, float mass) {
	// A(n)	= (-k * x(x) - b * v(n)) / mass
	// V(n+1) = V(n) + A(n) * delta_t
	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	Particle p = grid->get(x, y);
	p.position = sumd(p.position, sumd(scld(delta_t, pair.vel), scld(0.5f * delta_t * delta_t, pair.accel)));
	p.velocity = sumd(p.velocity, scld(delta_t, pair.accel));

	float diag_natural = natural * SQRT_2;
	
	// GetAccel
	for (int i = max(0, x - 1); i <= min(x + 1, m_row); i++) {
		for (int j = max(0, y - 1); j <= min(y + 1, m_col); j++) {
			if (!(i == x && j == y)) {
				Particle o = grid->get(i, j);
				o.position = sumd(o.position, sumd(scld(delta_t, o.velocity), scld(0.5f * delta_t * delta_t, o.acceleration)));
				o.velocity = sumd(o.velocity, scld(delta_t, o.acceleration));
				applyElasticForce(&p, &o, k, b, isDiag(x, y, i, j) ? diag_natural : natural);
			}
		}
	}

	return VelAccel{
		p.velocity,			//vel
		divd(p.force, mass)	//accel
	};
}

__global__ void rk4(Grid<Particle> * grid, float delta_t, int skip_x, int skip_y, float k, float b, float natural, float mass, float gravity) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;

	Particle p = grid->get(x, y);

	VelAccel init = { 0, 0 };
	VelAccel k1 = calcVel(grid, init, x, y, 0, k, b, natural, mass);
	VelAccel k2 = calcVel(grid, k1, x, y, 0.5f * delta_t, k, b, natural, mass);
	VelAccel k3 = calcVel(grid, k2, x, y, 0.5f * delta_t, k, b, natural, mass);
	VelAccel k4 = calcVel(grid, k3, x, y, delta_t, k, b, natural, mass); 

	// RK4: x(n+1) = x(n) + (k1 + 2 * (k2 + k3) + k4) delta_t/ 6
	p.position = sumd(p.position, scld(delta_t / 6, sumd(k1.vel, sumd(scld(2, sumd(k2.vel, k3.vel)), k4.vel))));
	p.velocity = sumd(p.velocity, scld(delta_t / 6, sumd(k1.accel, sumd(scld(2, sumd(k2.accel, k3.accel)), k4.accel))));
	p.force = { 0, 0, -gravity*p.mass };
	__syncthreads();
	grid->set(x, y, p);
	return;
}

__global__ void initialVelAccel(Grid<Particle> * grid, float delta_t, int skip_x, int skip_y, float k, float b, float natural, float mass, float gravity) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, skip_x, skip_y))
		return;
	VelAccel init = { 0, 0 };
	VelAccel currentVelAccel = calcVel(grid, init, x, y, 0, k, b, natural, mass);
	Particle p = grid->get(x, y);
	p.acceleration = currentVelAccel.accel;
	__syncthreads();
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
	
	// Velocity verlet
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

	// Traditional Verlet
	// x(n+1) = 2 * x(n) - x(n - 1) + a(n) * delta_t^2
	//Vec3 nextPos = p.position;
	//scl(&nextPos, 2);							// currPos term
	//scl(&p.prev_pos, -1);						// prevPos term
	//scl(&p.acceleration, delta_t * delta_t);	// accel term
	//sumf(&p.prev_pos, &nextPos);
	//sumf(&p.acceleration, &nextPos);

	//p.prev_pos = p.position;
	//p.position = nextPos;
	grid->set(x, y, p);
}

__global__ void interactGridAndParticle(Grid<Particle> * grid, Particle * big, int idx, int start_x, int start_y, int end_x, int end_y, float kn, float kt) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < 0 || y < 0 || x + start_x >= grid->getRows() || y + start_y >= grid->getCols())
		return;

	Particle p = grid->get(x + start_x, y + start_y);

	Vec3 p_pos = p.position;
	Vec3 o_pos = big[idx].position;

	float dist = distance(&p_pos, &o_pos);

	if (dist >= big[idx].radius + p.radius)
		return;

	float exn = (o_pos.x - p_pos.x) / dist;
	float eyn = (o_pos.y - p_pos.y) / dist;
	float ezn = (o_pos.z - p_pos.z) / dist;

	Vec3 normalForce = { exn, eyn, ezn };
	Vec3 tangentForce = { -ezn, -eyn, exn };

	float overlap = p.radius + big[idx].radius - dist;

	float normalForceMag = (-1) * kn * overlap;

	Vec3 p_vel = p.velocity;
	Vec3 o_vel = big[idx].velocity;
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
	atomic_sumf(&normalForce, &(big[idx].force));
	atomic_sumf(&tangentForce, &(big[idx].force));

	grid->set(x + start_x, y + start_y, p);

}



//This this assumes that particles are in a line
__global__ void interactBigParticles(Particle * particles, int size, float natural, float k, float b) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x < 0 || x >= size)
		return;

	Particle p = particles[x];

	if (x != 0) {
		Particle * left = &particles[x - 1];
		applyElasticForce(&p, left, k, b, natural);
	}
	else if (size > 2) {
		Particle * end = &particles[size - 1];
		applyElasticForce(&p, end, k, b, natural*(size - 1));
	}

	if (x != size - 1) {
		Particle * right = &particles[x + 1];
		applyElasticForce(&p, right, k, b, natural);
	} else if (size > 2) {
		Particle * begin = &particles[0];
		applyElasticForce(&p, begin, k, b, natural*(size - 1));
	}

	particles[x] = p;
}

__global__ void resetBigParticles(Particle * particles, int size, float g) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x < 0 || x >= size)
		return;

	Particle p = particles[x];

	p.force = { 0 ,0 , (-g) * p.mass };

	particles[x] = p;
}


__global__ void updateEulerBigParticles(Particle * particles, int size, float delta_t) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x < 0 || x >= size)
		return;

	Particle p = particles[x];

	Vec3 delta_v = { p.force.x / p.mass, p.force.y / p.mass, p.force.z / p.mass };
	scl(&delta_v, delta_t);
	sumf(&delta_v, &p.velocity);

	Vec3 delta_x = { p.velocity.x, p.velocity.y, p.velocity.z };
	scl(&delta_x, delta_t);
	sumf(&delta_x, &p.position);

	particles[x] = p;
}

__global__ void computeGridEnergy(Grid<Particle> * grid, float k, float natural, float * kinetic_t, float * elastic_t) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int m_row = grid->getRows() - 1;
	int m_col = grid->getCols() - 1;
	//Borders are fixed
	if (!checkGridBounds(x, y, m_row, m_col, 1, 1))
		return;

	Particle p = grid->get(x, y);

	float diag_natural = natural * SQRT_2;

	float kinetic = (1.0 / 2.0)*p.mass*dotf(&p.velocity, &p.velocity);
	float elastic = 0;
	for (int i = x - 1; i <= x + 1; i++) {
		for (int j = y - 1; j <= y + 1; j++) {
			if (!(i == x && j == y)) {
				Particle o = grid->get(i, j);
				float dst = distance(&p.position, &o.position);
				elastic += (1.0 / 2.0)*k*powf(dst - (isDiag(x, y, i, j) ? diag_natural : natural), 2);
			}
		}
	}

	atomicAdd(kinetic_t, kinetic);
	atomicAdd(elastic_t, elastic);

}

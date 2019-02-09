#pragma once
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.hpp"



struct Particle {
	Vec3 position = {}, velocity, force, acceleration = {}, prev_pos = {};
	float mass;
	float radius;
};

Particle * newParticles(Vec3 initial_position, float mass, float radius, int size, float separation) {
	Particle * ans = (Particle *)malloc(sizeof(Particle)*size);

	for (int i = 0; i < size; i++) {

		Particle p = ans[i];

		p.position = { initial_position.x + separation * i , initial_position.y, initial_position.z };

		p.velocity = { 0,0,0 };
		p.force = { 0,0,0 };

		p.radius = radius;
		p.mass = mass;

		ans[i] = p;
	}

	ans->prev_pos = ans->position;
	ans->acceleration = { 0, 0, 0 };

	return ans;
}

__host__ void updateParticleEuler(Particle * p_host, float delta_t) {

	Vec3 delta_v = { p_host->force.x / p_host->mass, p_host->force.y / p_host->mass, p_host->force.z / p_host->mass };
	scl(&delta_v, delta_t);
	sumf(&delta_v, &p_host->velocity);

	Vec3 delta_x = { p_host->velocity.x, p_host->velocity.y, p_host->velocity.z };
	scl(&delta_x, delta_t);
	sumf(&delta_x, &p_host->position);


}

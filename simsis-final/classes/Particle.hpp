#pragma once
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.hpp"



struct Particle {
	Vec3 position = {}, velocity, force, acceleration = {};
	float mass;
	float radius;
};

Particle * newParticle(Vec3 initial_position, float mass, float radius) {
	Particle * ans = (Particle *)malloc(sizeof(Particle));
	ans->position.x = initial_position.x;
	ans->position.y = initial_position.y;
	ans->position.z = initial_position.z;

	ans->velocity = { 0,0,0 };
	ans->force = { 0,0,0 };

	ans->radius = radius;
	ans->mass = mass;

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

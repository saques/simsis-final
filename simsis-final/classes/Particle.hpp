#pragma once
#include "Vector.hpp"

struct Particle {
	Vec3 position, velocity, force;
	float mass;
	float radius;
};

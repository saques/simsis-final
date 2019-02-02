#pragma once
#include "Vector.hpp"

struct Particle {
	Vec3 position = {}, velocity, force, acceleration = {};
	float mass;
	float radius;
};

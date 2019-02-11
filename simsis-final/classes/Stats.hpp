#pragma once
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particle.hpp"
#include "Vector.hpp"

class Stats {

public:

	//Rigid body stats
	float COM_height, deformation, big_energy;

	//Grid stats
	float grid_energy;

	__host__ __device__ void setCOM_height(Particle * particles, int size) {
		COM_height = 0;
		float total_mass = 0;

		for (int i = 0; i < size; i++) {
			Particle p = particles[i];
			total_mass += p.mass;
			COM_height += p.mass*p.position.z;
		}
		COM_height /= total_mass;
	}

	__host__ __device__ void setDeformation(Particle * particles, int size, float natural_distance) {
		if (size == 1) {
			deformation = 1;
			return;
		}
		Particle p = particles[0];
		Particle q = particles[size - 1];
		float actual = distance(&p.position, &q.position);
		deformation = (natural_distance*(size-1)) / actual;
	}

	__host__ __device__ void resetBig_energy() {
		big_energy = 0;
	}

	__host__ __device__ void addBig_energy(Particle * particles, int size, float g) {
		for (int i = 0; i < size; i++) {
			Particle p = particles[i];
			float gravitational = p.mass*g*p.position.z;
			float kinetic = (1.0f / 2.0f)*p.mass*dotf(&p.velocity, &p.velocity);
			big_energy += (gravitational + kinetic);
		}
	}

};
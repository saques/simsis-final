#pragma once
#include "cuda_runtime.h"

struct Vec3 {
	float x, y, z;
};

__global__ void  dot(Vec3*, Vec3*, float*);
__global__ void  nor(Vec3*, Vec3*, float*);
__global__ void sum(const Vec3*, const Vec3*, Vec3*);
__global__ void diff(Vec3*, Vec3*, Vec3*);
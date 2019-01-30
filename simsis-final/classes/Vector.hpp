#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

struct Vec3 {
	float x, y, z;
};

__global__ void sum(const Vec3* a, const Vec3* b, Vec3* c) {
	int i = threadIdx.x;
	c[i].x = a[i].x + b[i].x;
	c[i].y = a[i].y + b[i].y;
	c[i].z = a[i].z + b[i].z;
}

__global__ void diff(Vec3* a, Vec3* b, Vec3* c) {
	int i = threadIdx.x;
	c[i].x = a[i].x - b[i].x;
	c[i].y = a[i].y - b[i].y;
	c[i].z = a[i].z - b[i].z;
}

__global__ void dot(Vec3* a, Vec3* b, float* s) {
	int i = threadIdx.x;
	s[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
}

__global__ void nor(Vec3* a, float* s) {
	int i = threadIdx.x;
	s[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y + a[i].z * a[i].z);
}
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

struct Vec3 {
	float x, y, z;
};

__host__ __device__ float dotf(const Vec3* a, const Vec3* b) {
	return a->x * b->x + a->y * b->y + a->z * b->z;
}

__host__ __device__ float distance(const Vec3* a, const Vec3* b) {
	const Vec3 dif{ a->x - b->x , a->y - b->y, a->z - b->z };
	return sqrtf(dotf(&dif, &dif));
}

__host__ __device__ void rel(const Vec3* a, const Vec3* b, Vec3* c, float mod) {
	c->x = (a->x - b->x) / mod;
	c->y = (a->y - b->y) / mod;
	c->z = (a->z - b->z) / mod;
}

//b <- a+b
__host__ __device__ void sumf(const Vec3* a, Vec3* b) {
	b->x += a->x;
	b->y += a->y;
	b->z += a->z;
}

__host__ __device__ void scl(Vec3* a, float c) {
	a->x *= c;
	a->y *= c;
	a->z *= c;
}

__host__ __device__ void difff(const Vec3* a, const Vec3* b, Vec3* c) {
	c->x = a->x - b->x;
	c->y = a->y - b->y;
	c->z = a->z - b->z;
}

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
#pragma once
#include "cuda_runtime.h"

//The following kernels are just a demo
//for working with classes in CUDA/C++
__global__ void test(int * i) {

	Grid<int> * g = new Grid<int>(1, 1);
	g->set(0, 0, 1);
	*i = g->get(0, 0);
	delete g;

}
__global__ void change(Grid<int> * g, int row, int col, int v) {
	g->set(row, col, v);
}

#pragma once
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HANDLE_CUDA_ERROR(ERROR) if ((ERROR) != cudaSuccess) { \
		fprintf(stderr, "Cuda memory error"); \
		return nullptr; \
		} \


template <class T>
class Grid {

private:

		

public:
	T * data;
	int rows, cols;
	
	__host__ __device__ int getRows() { return rows; }

	__host__ __device__ int getCols() { return cols; }

	__host__ __device__ Grid<T>(int rows_, int cols_) : rows(rows_), cols(cols_) {

		int size = sizeof(T)*rows*cols;

		#ifndef __CUDACC__

		cudaMalloc((void **)&data, size);

		#else // !__CUDACC__

		data = (T*)malloc(size);

		#endif
	}

	__host__ __device__ Grid<T>(void){
		data = NULL;
	}

	__host__ __device__ ~Grid<T>() {
		#ifndef __CUDACC__

		cudaFree(data);

		#else // !__CUDACC__

		free(data);

		#endif
	}

	__host__ __device__ bool isOutOfBounds(int row, int col) {
		return !(row >= 0 && row < rows && col >= 0 && col < cols);
	}

	__host__ __device__ void set(int row, int col, T v) {
		if (!isOutOfBounds(row, col)) {
			data[row*cols + col] = v;
		}
	}

	__host__ __device__ T get(int row, int col) {
		return data[row*cols + col];
	}
	__host__ __device__ T*  getRef(int row, int col) {
		return &data[row*cols + col];
	}

	static enum copymode {
		UPLOAD, DOWNLOAD
	};

    __host__ static Grid<T> * gridcpy(const Grid<T> * grid, copymode mode) {
		Grid<T> * ans;
		Grid<T> tmp;
		int size;

		switch (mode) {
			case UPLOAD: 

				cudaMalloc((void **)&ans, sizeof(Grid<T>));


				//Copy grid data from host to device
				T * ans_data_dst;
				size = sizeof(T)*(grid->rows)*(grid->cols);
				cudaMalloc((void **)&(ans_data_dst), size);
				cudaMemcpy(ans_data_dst, grid->data, size, cudaMemcpyHostToDevice);

				//Assign data to structure and copy to device
				tmp.data = ans_data_dst;
				tmp.rows = grid->rows;
				tmp.cols = grid->cols;
				cudaMemcpy(ans, &tmp, sizeof(Grid<T>), cudaMemcpyHostToDevice);
				tmp.data = NULL;


				break;
			case DOWNLOAD: 

				ans = (Grid<T>*)malloc(sizeof(Grid<T>));
				HANDLE_CUDA_ERROR(cudaMemcpy(ans, grid, sizeof(Grid<T>), cudaMemcpyDeviceToHost));

				//Obtain pointer in device from copyied data
				T * data_device_ptr;
				data_device_ptr = ans->data;

				//Allocate memory in host for grid data
				size = sizeof(T)*(ans->rows)*(ans->cols);
				ans->data = (T*)malloc(size);

				//Copy data back to host
				HANDLE_CUDA_ERROR(cudaMemcpy(ans->data, data_device_ptr, size, cudaMemcpyDeviceToHost));

				break;

			default: break;
		}


		return ans;
	}

};



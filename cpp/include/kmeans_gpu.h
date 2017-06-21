// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Austin Clyde

#ifndef CPP_INCLUDE_KMEANS_GPU_H_
#define CPP_INCLUDE_KMEANS_GPU_H_

#ifdef CUDA_AND_GPU

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <string>

#include "Eigen/Dense"
#include "Eigen/Core"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"
#include "include/kmeans.h"

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif


namespace Nice {

float **cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                    int numCoords,    /* no. features */
                    int numObjs,      /* no. objects */
                    int numClusters,  /* no. clusters */
                    float threshold,    /* % objects change membership */
                    int *membership,   /* out: [numObjs] */
                    int *loop_iterations);

template<typename T>
using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor> >;


template<typename T>
class KMeansGPU : public KMeans<T> {
 public:
  KMeansGPU() {

  }

  ~KMeansGPU() {

  }
  KMeansGPU(const KMeansGPU &rhs) {}

  void Fit(const Matrix<T> &input_data, int k) {
    float **centers;
    int *labels;
    float threshold = 0.00001;
    int max_iter = 10000;
    int numSamples = input_data.rows(), numFeatures = input_data.cols();
    int *membership = (int*) malloc(numSamples * sizeof(int));

    //eigen to 2d array [samples][features]
    float **objects = (float**) malloc(numSamples * sizeof(float*));
    for (int m = 0; m < numSamples; m++) {
      objects[m] = (float *) malloc(numFeatures * sizeof(float));
      for (int n = 0; n < numFeatures; n++) {
        objects[m][n] = input_data(m,n);
        }
    }

    centers = cuda_kmeans(objects, numFeatures, numSamples, k, threshold, membership, &max_iter);
    labels = membership;

    Eigen::MatrixXf clusters(k, numFeatures);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < numFeatures; j++)
        clusters(i,j) = centers[i][j];
    }

    KMeans<T>::centers_ = clusters.transpose();
  }
 private:


}; // Class KMMEANS_GPU
  template class KMeans<float>;

}  // namespace Nice
#endif  // CUDA_AND_GPU
#endif  // CPP_INCLUDE_KMEANS_GPU_H_
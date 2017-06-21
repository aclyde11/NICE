//
// Created by Austin Clyde on 6/21/17.
//

#ifndef NICE_KMEANS_H
#define NICE_KMEANS_H



#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include "include/matrix.h"
#include "include/vector.h"


namespace Nice {


template<typename T>
class KMeans {
 public:

  KMeans() {

  }

  ~KMeans() {

  }

  virtual void Fit(const Matrix <T> &input_data, int k) { };

  virtual void SetRandom(const bool r) {
    this->random_ = r;
  }

  virtual Matrix <T> GetLabels() {
    return labels_;
  }

  virtual Matrix <T> GetCenters() {
    return centers_;
  }

 protected:
  Vector<T> labels_;
  bool random_ = true;
  unsigned int k_;
  Matrix<T> centers_;
};
}
#endif //NICE_KMEANS_H

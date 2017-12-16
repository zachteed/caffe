// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/nearest_neighbor_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void NearestNeighborForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int top_height, const int top_width, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int w = index % top_width;
    int h = (index / top_width) % top_height;
    int c = (index / top_width / top_height) % channels;
    int n = index / top_width / top_height / channels;

    int bottom_index = (n*channels + c) * height * width;
    bottom_index += (h/2) * width + (w/2);
    top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
void NearestNeighborLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  NearestNeighborForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_, height_, width_, top_height_, top_width_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void NearestNeighborBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height, const int width,
    const int top_height, const int top_width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;

    int top_index = (n*channels+c)*top_height*top_width;
    const Dtype *offset_top_diff = top_diff + top_index;


    gradient += offset_top_diff[2*h*top_width + 2*w];
    gradient += offset_top_diff[(2*h+1)*top_width + 2*w];
    gradient += offset_top_diff[2*h*top_width + (2*w+1)];
    gradient += offset_top_diff[(2*h+1)*top_width + (2*w+1)];
    bottom_diff[index] = gradient;

  }
}

template <typename Dtype>
void NearestNeighborLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  // NOLINT_NEXT_LINE(whitespace/operators)
  NearestNeighborBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, channels_, height_, width_,
      top_height_, top_width_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NearestNeighborLayer);

}  // namespace caffe

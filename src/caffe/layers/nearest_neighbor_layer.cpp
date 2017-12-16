#include <cfloat>

#include "caffe/layers/nearest_neighbor_layer.hpp"


using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void NearestNeighborLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NearestNeighborParameter nearest_neighbor_param = this->layer_param_.nearest_neighbor_param();
}

template <typename Dtype>
void NearestNeighborLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top_height_ = 2 * height_;
  top_width_ = 2 * width_;
  top[0]->Reshape(bottom[0]->num(), channels_, top_height_, top_width_);
}


template <typename Dtype>
void NearestNeighborLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();


  for (int n=0; n<top[0]->num(); n++) {
    for (int c=0; c<top[0]->channels(); c++) {
      for (int h=0; h<top[0]->height(); h++) {
        for (int w=0; w<top[0]->width(); w++) {
          int index = (h/2)*width_ + w/2;
          top_data[h*top_width_+w] = bottom_data[index];
        }
      }
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void NearestNeighborLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(NearestNeighborLayer);
#endif

INSTANTIATE_CLASS(NearestNeighborLayer);
REGISTER_LAYER_CLASS(NearestNeighbor);

}  // namespace caffe

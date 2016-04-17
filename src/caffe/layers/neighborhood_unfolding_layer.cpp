#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NeighborhoodUnfoldingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeighborhoodUnfoldingParameter params = this->layer_param_.neighborhood_unfolding_param();
  neigh_h_ = params.neighborhood_h();
  neigh_w_ = params.neighborhood_w();
  pad_h_ = neigh_h_/2;//params.pad_h();
  pad_w_ = neigh_w_/2;//params.pad_w();
  copy_padding_ = params.copy_padding();
  self_copying_ = params.self_copying();

}

template <typename Dtype>
void NeighborhoodUnfoldingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  im_h_ = bottom[0]->height();
  im_w_ = bottom[0]->width();
  im_unfolded_h_ = im_h_ * neigh_h_;
  im_unfolded_w_ = im_w_ * neigh_w_;
  top[0]->Reshape(bottom[0]->num(), channels_, im_unfolded_h_, im_unfolded_w_);
}

template <typename Dtype>
void NeighborhoodUnfoldingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const Dtype* im = bottom[0]->cpu_data() + channels_ * im_h_ * im_w_ * n;
    Dtype* im_unfolded = top[0]->mutable_cpu_data() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;

    for (int c = 0; c < channels_; ++c){
      for (int i = 0; i < im_h_; ++i){
        for (int j = 0; j < im_w_; ++j){
          for (int k = 0; k < neigh_h_; ++k){
            for (int l = 0; l < neigh_w_; ++l){
              int im_unfolded_i = i * neigh_h_ + k;
              int im_unfolded_j = j * neigh_w_ + l; 
              int im_i = i + k - neigh_h_/2;
              int im_j = j + l - neigh_w_/2;
            
              if (copy_padding_){
                if (im_i < 0) 
                  im_i += pad_h_;
                if (im_j < 0)
                  im_j += pad_w_;
                if (im_i >= im_h_) 
                  im_i -= pad_h_;
                if (im_j >= im_w_) 
                  im_j -= pad_w_;  
              }  
              int index = im_unfolded_w_ * im_unfolded_h_ * c + im_unfolded_i * im_unfolded_w_ + im_unfolded_j;             
            
              if ((im_i < 0) || (im_i >= im_h_) || (im_j < 0) || (im_j >= im_w_)) 
                im_unfolded[index] = 0;
              else { 
                if (self_copying_)
                  im_unfolded[index] = im[im_w_ * im_h_ * c + i * im_w_ + j];
                else   
                  im_unfolded[index] = im[im_w_ * im_h_ * c + im_i * im_w_ + im_j];
              } 
            }     
          }
        }
      }
    }
  }              
}

template <typename Dtype>
void NeighborhoodUnfoldingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;

  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  for (int n = 0; n < bottom[0]->num(); ++n) {
    Dtype* im = bottom[0]->mutable_cpu_diff() + channels_ * im_h_ * im_w_ * n;
    const Dtype* im_unfolded = top[0]->cpu_diff() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;
     
    for (int c = 0; c < channels_; ++c){
      for (int i = 0; i < im_h_; ++i){
        for (int j = 0; j < im_w_; ++j){
          for (int k = 0; k < neigh_h_; ++k){
            for (int l = 0; l < neigh_w_; ++l){
              int im_unfolded_i = i * neigh_h_ + k;
              int im_unfolded_j = j * neigh_w_ + l; 
              int im_i = i + k - neigh_h_/2;
              int im_j = j + l - neigh_w_/2;
            
              if (copy_padding_){
                if (im_i < 0) 
                  im_i += pad_h_;
                if (im_j < 0)
                  im_j += pad_w_;
                if (im_i >= im_h_) 
                  im_i -= pad_h_;
                if (im_j >= im_w_) 
                  im_j -= pad_w_;  
              }  
              int index = im_unfolded_w_ * im_unfolded_h_ * c + im_unfolded_i * im_unfolded_w_ + im_unfolded_j;             
            
              if (((im_i >= 0) && (im_i < im_h_) && (im_j >= 0) && (im_j < im_w_)) ){
                if (self_copying_)
                  im[im_w_ * im_h_ * c + i * im_w_ + j] += im_unfolded[index];
                else   
                  im[im_w_ * im_h_ * c + im_i * im_w_ + im_j] += im_unfolded[index];
              } 
            }     
          }
        }
      }
    }
  }         
}

#ifdef CPU_ONLY
STUB_GPU(NeighborhoodUnfoldingLayer);
#endif

INSTANTIATE_CLASS(NeighborhoodUnfoldingLayer);
REGISTER_LAYER_CLASS(NeighborhoodUnfolding);

}  // namespace caffe

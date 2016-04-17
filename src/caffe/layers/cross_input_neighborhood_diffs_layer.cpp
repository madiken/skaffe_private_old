#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossInputNeighborhoodDiffsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeighborhoodUnfoldingParameter params = this->layer_param_.neighborhood_unfolding_param();
  neigh_h_ = params.neighborhood_h();
  neigh_w_ = params.neighborhood_w();
  pad_h_ = neigh_h_/2;//params.pad_h();
  pad_w_ = neigh_w_/2;//params.pad_w();
  copy_padding_ = params.copy_padding();


}

template <typename Dtype>
void CrossInputNeighborhoodDiffsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());


  channels_ = bottom[0]->channels();
  im_h_ = bottom[0]->height();
  im_w_ = bottom[0]->width();
  im_unfolded_h_ = im_h_ * neigh_h_;
  im_unfolded_w_ = im_w_ * neigh_w_;
  top[0]->Reshape(bottom[0]->num(), channels_, im_unfolded_h_, im_unfolded_w_);
  top[1]->Reshape(bottom[0]->num(), channels_, im_unfolded_h_, im_unfolded_w_);
}

template <typename Dtype>
void CrossInputNeighborhoodDiffsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  for (int n = 0; n < bottom[0]->num(); ++n) {
    const Dtype* im0 = bottom[0]->cpu_data() + channels_ * im_h_ * im_w_ * n;
    const Dtype* im1 = bottom[1]->cpu_data() + channels_ * im_h_ * im_w_ * n;
    
    Dtype* cross_input_diff0 = top[0]->mutable_cpu_data() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;
    Dtype* cross_input_diff1 = top[1]->mutable_cpu_data() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;

  
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
              
              if ((im_i < 0) || (im_i >= im_h_) || (im_j < 0) || (im_j >= im_w_)) {
                  //actually we could still count copying pixels, but just zeroing is simpler
                  cross_input_diff0[index] = 0;
                  cross_input_diff1[index] = 0;
              } else { 
                
                cross_input_diff0[index] = im0[im_w_ * im_h_ * c + i * im_w_ + j] - im1[im_w_ * im_h_ * c + im_i * im_w_ + im_j];
                cross_input_diff1[index] = im1[im_w_ * im_h_ * c + i * im_w_ + j] - im0[im_w_ * im_h_ * c + im_i * im_w_ + im_j];  

              } 
            }     
          }
        }
      }
    }
  }              
}

template <typename Dtype>
void CrossInputNeighborhoodDiffsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;

  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  for (int n = 0; n < bottom[0]->num(); ++n) {

    Dtype* im0 = bottom[0]->mutable_cpu_diff() + channels_ * im_h_ * im_w_ * n;
    Dtype* im1 = bottom[1]->mutable_cpu_diff() + channels_ * im_h_ * im_w_ * n;
    
    const Dtype* cross_input_diff0 = top[0]->cpu_diff() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;
    const Dtype* cross_input_diff1 = top[1]->cpu_diff() + channels_ * im_unfolded_h_ * im_unfolded_w_ * n;

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
                im0[im_w_ * im_h_ * c + i * im_w_ + j] += cross_input_diff0[index];
                im1[im_w_ * im_h_ * c + i * im_w_ + j] += cross_input_diff1[index];

                im0[im_w_ * im_h_ * c + im_i * im_w_ + im_j] -= cross_input_diff1[index];
                im1[im_w_ * im_h_ * c + im_i * im_w_ + im_j] -= cross_input_diff0[index];
              } 
            }     
          }
        }
      }
    }
  }         
}

#ifdef CPU_ONLY
STUB_GPU(CrossInputNeighborhoodDiffsLayer);
#endif

INSTANTIATE_CLASS(CrossInputNeighborhoodDiffsLayer);
REGISTER_LAYER_CLASS(CrossInputNeighborhoodDiffs);

}  // namespace caffe

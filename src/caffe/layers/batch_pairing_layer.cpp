#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {


template <typename Dtype>
void BatchPairingLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // CHECK_EQ(bottom[0]->height(), 1);
  // CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);

}
         
template <typename Dtype>
void BatchPairingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int top_num = bottom[0]->num()*(bottom[0]->num() - 1)/2;
  top[0]->Reshape(top_num, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()); //1
  top[1]->Reshape(top_num, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()); //2
  top[2]->Reshape(top_num, 1, 1, 1);                     //label
  num_ = bottom[0]->num();
}

template <typename Dtype>
void BatchPairingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int pos_label = this->layer_param_.batch_pairing_param().pos_label(); 
  int neg_label = this->layer_param_.batch_pairing_param().neg_label(); 
//  std::cout << "pos_label " << pos_label << " neg_label " << neg_label << std::endl;
  int n1 = 0;
  int n2 = 0; 	

  int k = 0;
  for (int i = 0; i < num_; ++i) {
    for (int j = i+1; j < num_; ++j) {
      //top[0]->mutable_cpu_data()[k] = xy_[i][j]/sqrt(xy_[i][i] * xy_[j][j]);
      caffe_copy(channels * height * width, bottom[0]->cpu_data() + i*channels*height*width, top[0]->mutable_cpu_data() + k*channels*height*width);
      caffe_copy(channels * height * width, bottom[0]->cpu_data() + j*channels*height*width, top[1]->mutable_cpu_data() + k*channels*height*width);

      if (bottom[1]->cpu_data()[i] == bottom[1]->cpu_data()[j]){
        top[2]->mutable_cpu_data()[k++] = pos_label;
        n1++;
      }else{
        top[2]->mutable_cpu_data()[k++] = neg_label;
        n2++;
      } 	
    }
  }

  //LOG(INFO)<< "n1 " << n1 << " n2 " << n2;  
}

template <typename Dtype>
void BatchPairingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
  if (!propagate_down[0])
    return;
      
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int num = bottom[0]->num();

  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

  for (int i = 0; i < num; ++i) {
   // for (int k = 0; k < channels; ++k) {
     // Dtype dsdx = 0;
      for (int j = 0; j < num; ++j) {
	//i j --> h    
        int h;
        int pair_index;
        if (i < j){
          pair_index = 0;
          h =  num * i - i * (i+1)/2 + j-i-1;
        } else if (i > j) {
          pair_index = 1;
          h =  num * j - j * (j+1)/2 + i-j-1;               
        } else continue;
        //Dtype add_dsdx = top[pair_index]->cpu_diff()[h*channels*height*width + k]; 
        //dsdx += add_dsdx;
        caffe_axpy(channels * height * width, Dtype(1.), top[pair_index]->cpu_diff() + h*channels*height*width, bottom[0]->mutable_cpu_diff() + i*channels*height*width);
      }		
   //   bottom[0]->mutable_cpu_diff()[i*channels*height*width + k] = dsdx; 
   // }
  } 
}

#ifdef CPU_ONLY
STUB_GPU(BatchPairingLayer);
#endif

INSTANTIATE_CLASS(BatchPairingLayer);
REGISTER_LAYER_CLASS(BatchPairing);
}  // namespace caffe

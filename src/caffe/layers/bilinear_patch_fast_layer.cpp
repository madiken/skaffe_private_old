#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {


template <typename Dtype>
void BilinearPatchFastLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  CHECK_EQ(bottom[0]->width()*bottom[0]->height(),bottom[1]->width()*bottom[1]->height());

  CHECK_EQ(bottom[2]->width()*bottom[2]->height(),bottom[1]->width()*bottom[1]->height());

  poolingFieldsNum = bottom[2]->channels();

  masked_buffer1.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  masked_buffer2.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());


  dlda_buffer.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  dldb_buffer.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
}
     
template <typename Dtype>
void multiplyAllChannelsByMask(const Dtype* blob, const Dtype*  mask_blob, int mask_num, Dtype* blob_result, int sz, int blob_channels){
  int data_offset = 0;
  int mask_offset = mask_num * sz;

    for(int j = 0; j < blob_channels; j++){
      data_offset = j * sz;      
      caffe_mul(sz, blob + data_offset, mask_blob + mask_offset, blob_result + data_offset);
    }
}
    
template <typename Dtype>
void BilinearPatchFastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 // outer - positions, inner - channels
  top[0]->Reshape(bottom[0]->num(),  poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels(), 1, 1);   
}

template <typename Dtype>
void BilinearPatchFastLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  

  for (int n = 0; n < bottom[0]->num(); n++){
    for (int i = 0; i < poolingFieldsNum; i++){
       multiplyAllChannelsByMask(bottom[0]->cpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i, masked_buffer1.mutable_cpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

       multiplyAllChannelsByMask(bottom[1]->cpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i, masked_buffer2.mutable_cpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.cpu_data(), masked_buffer2.cpu_data(), (Dtype)0., top[0]->mutable_cpu_data() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels());


    }
  }
}

template <typename Dtype>
void BilinearPatchFastLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_set(bottom[0]->num()*bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->num()*bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), Dtype(0.0), bottom[1]->mutable_cpu_diff());


  for (int n = 0; n < bottom[0]->num(); n++){

    for(int i = 0; i < poolingFieldsNum; i++){
      if (propagate_down[0]) {
        
        multiplyAllChannelsByMask(bottom[1]->cpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i, masked_buffer2.mutable_cpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., top[0]->cpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.cpu_data(), (Dtype)0., dlda_buffer.mutable_cpu_diff());
	
	
	multiplyAllChannelsByMask(dlda_buffer.cpu_diff(), bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i,dlda_buffer.mutable_cpu_diff(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

        caffe_add(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), dlda_buffer.cpu_diff(), bottom[0]->cpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[0]->mutable_cpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n);

      }
	
      if (propagate_down[1]) {

         multiplyAllChannelsByMask(bottom[0]->cpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i, masked_buffer1.mutable_cpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());
        
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., top[0]->cpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.cpu_data(), (Dtype)0., dldb_buffer.mutable_cpu_diff());


	multiplyAllChannelsByMask(dldb_buffer.cpu_diff(), bottom[2]->cpu_data() + bottom[2]->channels() * bottom[2]->height() * bottom[2]->width() * n, i,dldb_buffer.mutable_cpu_diff(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_add(bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), dldb_buffer.cpu_diff(), bottom[1]->cpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[1]->mutable_cpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n);

      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(BilinearPatchFastLayer);
#endif

INSTANTIATE_CLASS(BilinearPatchFastLayer);
REGISTER_LAYER_CLASS(BilinearPatchFast);
}  // namespace caffe


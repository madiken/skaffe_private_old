#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 

#include <iostream>
#include <fstream>
namespace caffe {



template <typename Dtype>
__global__ void multiplyAllChannelsByMaskGpu(const Dtype* blob, const Dtype*  mask_blob, int mask_num, Dtype* blob_result, int sz, const int blob_channels){
  int data_offset = 0;
  int mask_offset = mask_num * sz;

  CUDA_KERNEL_LOOP(index, blob_channels*sz){
    //for(int j = 0; j < blob_channels; j++){
      int j = index / sz;
      data_offset = j * sz;      
      
       blob_result[data_offset + index % sz] = mask_blob[mask_offset + index % sz] * blob[data_offset + index % sz];
     // caffe_gpu_mul(sz, blob + data_offset, mask_blob + mask_offset, blob_result + data_offset);
   // }
  } 
}


/*
template <typename Dtype>
void multiplyAllChannelsByMaskGpu(const Dtype* blob, const Dtype*  mask_blob, int mask_num, Dtype* blob_result, int sz, const int blob_channels){
  int data_offset = 0;
  int mask_offset = mask_num * sz;

  CUDA_KERNEL_LOOP(j, blob_channels){
    //for(int j = 0; j < blob_channels; j++){
      data_offset = j * sz;      
      caffe_gpu_mul(sz, blob + data_offset, mask_blob + mask_offset, blob_result + data_offset);
   // }
  } 
}
*/

template <typename Dtype>
void BilinearPatchLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  

  for (int n = 0; n < bottom[0]->num(); n++){
    for (int i = 0; i < poolingFieldsNum; i++){
       multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->gpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.gpu_data(), i, masked_buffer1.mutable_gpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

       multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[1]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->gpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.gpu_data(), i, masked_buffer2.mutable_gpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.gpu_data(), masked_buffer2.gpu_data(), (Dtype)0., transpBuffer_top.mutable_gpu_data() + n * transpBuffer_top.channels()  + i * bottom[0]->channels() * bottom[1]->channels());

/*
caffe_gpu_geam_old<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
*/
       
    }

    caffe_gpu_geam_old(CblasNoTrans, CblasTrans, top[0]->channels(), top[0]->height() * top[0]->width(), (Dtype)0.0, top[0]->gpu_data() + n * top[0]->channels() * top[0]->width() * top[0]->height(), transpBuffer_top.gpu_data() + n * transpBuffer_top.channels(), (Dtype)1.0, top[0]->mutable_gpu_data() + n * top[0]->channels() * top[0]->width() * top[0]->height() );


  }
}

template <typename Dtype>
void BilinearPatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_set(bottom[0]->num()*bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->num()*bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), Dtype(0.0), bottom[1]->mutable_gpu_diff());


  for (int n = 0; n < bottom[0]->num(); n++){
  
    caffe_gpu_geam_old(CblasNoTrans, CblasTrans, top[0]->height() * top[0]->width(), top[0]->channels(), (Dtype)0.0, transpBuffer_top.gpu_diff() + n * transpBuffer_top.channels(),
top[0]->gpu_diff() + n * top[0]->channels() * top[0]->width() * top[0]->height(),  (Dtype)1.0, transpBuffer_top.mutable_gpu_diff() + n * transpBuffer_top.channels() );
    
    for(int i = 0; i < poolingFieldsNum; i++){
      if (propagate_down[0]) {
        
        multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[1]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->gpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.gpu_data(), i, masked_buffer2.mutable_gpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., transpBuffer_top.gpu_diff() + n * transpBuffer_top.channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.gpu_data(), (Dtype)0., dlda_buffer.mutable_gpu_diff());
	
	
	multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(dlda_buffer.gpu_diff(), mask_buffer.gpu_data(), i,dlda_buffer.mutable_gpu_diff(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

        caffe_gpu_add(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), dlda_buffer.gpu_diff(), bottom[0]->gpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[0]->mutable_gpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n);

      }
	
      if (propagate_down[1]) {

         multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->gpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.gpu_data(), i, masked_buffer1.mutable_gpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());
        
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., transpBuffer_top.gpu_diff() + n * transpBuffer_top.channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.gpu_data(), (Dtype)0., dldb_buffer.mutable_gpu_diff());


	multiplyAllChannelsByMaskGpu<<<CAFFE_GET_BLOCKS(bottom[1]->channels()*bottom[0]->height()*bottom[0]->width()), CAFFE_CUDA_NUM_THREADS>>>(dldb_buffer.gpu_diff(),mask_buffer.gpu_data(), i,dldb_buffer.mutable_gpu_diff(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_gpu_add(bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), dldb_buffer.gpu_diff(), bottom[1]->gpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[1]->mutable_gpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n);

      }
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearPatchLayer);
}  // namespace caffe


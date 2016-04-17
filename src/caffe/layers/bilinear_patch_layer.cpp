#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {

template <typename Dtype>
void  generateMasks(int sz_h, int sz_w, int patch_h, int patch_w, Dtype* dest){
    int num_h = (int) (sz_h / patch_h);
    int num_w = (int) (sz_w / patch_w);
    
    if (sz_h % patch_h > 0)
        num_h += 1;
    if (sz_w % patch_w > 0)
        num_w += 1;

    caffe_set(num_h*num_w*sz_h*sz_w, (Dtype)0.0, dest);

    for (int ii = 0; ii < sz_h; ii++){
        for (int jj = 0; jj < sz_w; jj++){
            int i = (int)ii/patch_h;
            int j = (int)jj/patch_w;
            
            int mask_num = i*num_w + j;
            dest[mask_num * sz_h * sz_w  + ii*sz_w + jj] = 1;
        }
    }
}

template <typename Dtype>
void BilinearPatchLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  patch_h = this->layer_param_.bilinear_patch_param().patch_h();
  patch_w = this->layer_param_.bilinear_patch_param().patch_w();
    int num_h = (int) (bottom[0]->height()/ patch_h);
    int num_w = (int) (bottom[0]->width()/ patch_w);
    
    if (bottom[0]->height() % patch_h > 0)
        num_h += 1;
    if (bottom[0]->width()% patch_w > 0)
        num_w += 1;

  poolingFieldsNum = num_h * num_w;
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
 // CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->width()*bottom[0]->height(),bottom[1]->width()*bottom[1]->height());
  //CHECK_EQ(bottom[2]->width()*bottom[2]->height(),bottom[1]->width()*bottom[1]->height());
 // CHECK_EQ(poolingFieldsNum,bottom[2]->channels()); 
  //CHECK_EQ(bottom[2]->num(), 1);

  masked_buffer1.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  masked_buffer2.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());


  dlda_buffer.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  dldb_buffer.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  transpBuffer_top.Reshape(bottom[0]->num(), poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels(), 1, 1);


  mask_buffer.Reshape(1, poolingFieldsNum, bottom[0]->height(), bottom[0]->width());

  generateMasks(bottom[0]->height(), bottom[0]->width(), patch_h, patch_w, mask_buffer.mutable_cpu_data());


  std::cout<< "##########################3################## " << std::endl;
  std::cout<< mask_buffer.channels() << " " << mask_buffer.height() << " "  << mask_buffer.width() << std::endl;

  for (int i = 0; i < mask_buffer.count(); i++){
     std::cout << mask_buffer.cpu_data()[i] << " ";
     if ((i+1) % ( mask_buffer.height() * mask_buffer.width()) == 0)
        std::cout << "---------------------------" << std::endl;
     if ((i+1) %  mask_buffer.width() == 0)
        std::cout << std::endl;

  } 

  
 
}
     

template <typename Dtype>
void BilinearPatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  
    int num_h = (int) (bottom[0]->height() / patch_h);
    int num_w = (int) (bottom[0]->width()/ patch_w);
    
    if (bottom[0]->height()% patch_h > 0)
        num_h += 1;
    if (bottom[0]->width() % patch_w > 0)
        num_w += 1;

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels() * bottom[1]->channels(), num_h, num_w); 


//  caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(), mask_buffer.mutable_cpu_data());   
  
}

template <typename Dtype>
void BilinearPatchLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 

   // NOT_IMPLEMENTED;
 // caffe_set(top[0]->num()*bottom[0]->channels()*top[0]->height()*top[0]->width(), Dtype(0.0), top[0]->mutable_cpu_data());
 
 }



template <typename Dtype>
void BilinearPatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //caffe_set(bottom[0]->num()*bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
  //caffe_set(bottom[1]->num()*bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), Dtype(0.0), bottom[1]->mutable_cpu_diff());

   // NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(BilinearPatchLayer);
#endif

INSTANTIATE_CLASS(BilinearPatchLayer);
REGISTER_LAYER_CLASS(BilinearPatch);
}  // namespace caffe


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class EltwiseProdGroupLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EltwiseProdGroupLayerTest()
      : blob_bottom_data_maps(new Blob<Dtype>(1, 15, 5, 5)),
        blob_bottom_data_masks(new Blob<Dtype>(1, 1, 5, 5)),
        blob_top(new Blob<Dtype>(1, 15, 5, 5))
  {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(0.1);
    filler_param.set_std(0.1); 
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_maps);
    blob_bottom_vec_.push_back(blob_bottom_data_maps);
    
    filler.Fill(this->blob_bottom_data_masks);
    blob_bottom_vec_.push_back(blob_bottom_data_masks);

    
    blob_top_vec_.push_back(blob_top);
    
  }

  virtual ~EltwiseProdGroupLayerTest() {
    delete blob_bottom_data_maps;
    delete blob_bottom_data_masks;
    delete blob_top;
   
  }

  Blob<Dtype>* const blob_bottom_data_maps;
  Blob<Dtype>* const blob_bottom_data_masks;
  Blob<Dtype>* const blob_top;
 
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EltwiseProdGroupLayerTest, TestDtypesAndDevices);

TYPED_TEST(EltwiseProdGroupLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseProdGroupLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare

  vector<Dtype> top_man;
  int num = this->blob_bottom_data_maps->num();
  int ch = this->blob_bottom_data_maps->channels();
  int he = this->blob_bottom_data_maps->height();
  int wi = this->blob_bottom_data_maps->width();
  Dtype map_d;
  Dtype mask_d;
  for(int i = 0; i < num; i++)
    for (int j = 0; j < ch; j++)
      for (int k = 0; k < he; k++)
        for(int l = 0; l < wi; l++){
          map_d = this->blob_bottom_data_maps->cpu_data()[i * ch * he * wi + j * he * wi + k * wi +l];
          mask_d = this->blob_bottom_data_masks->cpu_data()[i * he * wi  + k * wi +l];
          
          EXPECT_NEAR(this->blob_top->cpu_data()[i * ch * he * wi + j * he * wi + k * wi +l] , map_d * mask_d, 1e-6);
        }
   
}

TYPED_TEST(EltwiseProdGroupLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseProdGroupLayer<Dtype> layer(layer_param);
  
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // for(int k = 0; k < this->blob_top_vec_[0]->count(); k++) {
  //   checker.CheckGradientSingle(&layer, (this->blob_bottom_vec_),
  //   (this->blob_top_vec_), 0, 0,
  //   k, false); 
  //  }
  //check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
    (this->blob_top_vec_), 0);
}

}  // namespace caffe

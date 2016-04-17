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
class BatchPairingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BatchPairingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 10, 1, 2)),
        blob_bottom_labels_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_data1_(new Blob<Dtype>(55, 10, 1, 2)),
        blob_top_data2_(new Blob<Dtype>(55, 10, 1, 2)),
        blob_top_labels_(new Blob<Dtype>(55, 1, 1, 1)){
    
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.01); 
    
    GaussianFiller<Dtype> filler(filler_param); 
    
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_labels_);
    
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_labels_);
    blob_top_vec_.push_back(blob_top_data1_);
    blob_top_vec_.push_back(blob_top_data2_);
    blob_top_vec_.push_back(blob_top_labels_);
     
  }
  virtual ~BatchPairingLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data1_;
    delete blob_top_data2_;
    delete blob_bottom_labels_;
    delete blob_top_labels_;
  }
 
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data1_;
  Blob<Dtype>* const blob_top_data2_;
  Blob<Dtype>* const blob_bottom_labels_;
  Blob<Dtype>* const blob_top_labels_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BatchPairingLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchPairingLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  BatchPairingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
 
}


TYPED_TEST(BatchPairingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchPairingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);
}

}  // namespace caffe

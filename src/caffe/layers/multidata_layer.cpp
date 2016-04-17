#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>
#include <google/protobuf/io/coded_stream.h>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using google::protobuf::io::CodedInputStream;
using google::uint32;

typedef unsigned char uchar; 

namespace caffe {

bool GetNextDatum(CodedInputStream* stream, Datum* datum) {
  uint32 message_size;
  if (!stream->ReadLittleEndian32(&message_size)) {
    return false;
  }
  CodedInputStream::Limit limit = stream->PushLimit(message_size);
  datum->ParseFromCodedStream(stream);
  stream->PopLimit(limit);
}

template <typename Dtype>
MultidataLayer<Dtype>::~MultidataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultidataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top_size_ = top.size();

  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.multidata_param().backend()));
  db_->Open(this->layer_param_.multidata_param().source(), db::READ);

  cursor_.reset(db_->NewCursor(this->layer_param_.multidata_param().cursor()));

  // Check if we should randomly skip a few data points
  if (this->layer_param_.multidata_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.multidata_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      cursor_->Next();
    }
  }

  // Fill missing params with default values.
  if (this->layer_param_.multidata_param().force_encoded_color_size() == 0) {
    for (int top_index = 0; top_index < top.size(); ++top_index) {
      this->layer_param_.mutable_multidata_param()->
          add_force_encoded_color(false);
    }
  }
  if (this->layer_param_.multidata_param().transform_param_size() == 0) {
    for (int top_index = 0; top_index < top.size(); ++top_index) {
      this->layer_param_.mutable_multidata_param()->add_transform_param();
    }
  }

  // Create temporary buffers.
  this->prefetch_data_.resize(top_size_);
  for (int top_index = 0; top_index < top_size_; ++top_index) {
    this->prefetch_data_[top_index].reset(new Blob<Dtype>());
  }
  this->transformed_data_.resize(top_size_);
  for (int top_index = 0; top_index < top_size_; ++top_index) {
    this->transformed_data_[top_index].reset(new Blob<Dtype>());
  }

  // Read a data point, and use it to initialize the top blob.
  string binary_data = cursor_->value();
  Datum datum;
  CodedInputStream stream(reinterpret_cast<const uchar*>(binary_data.data()), 
                          binary_data.size());

  for (int top_index = 0; top_index < top_size_; ++top_index) {
    GetNextDatum(&stream, &datum);

    bool force_color = 
        this->layer_param_.multidata_param().force_encoded_color(top_index);

    int crop_size = 
        this->layer_param_.multidata_param().
        transform_param(top_index).crop_size();

    if ((force_color && DecodeDatum(&datum, true)) ||
        DecodeDatumNative(&datum)) {
      LOG(INFO) << "Decoding Datum";
    }
    if (crop_size > 0) {
      top[top_index]->Reshape(this->layer_param_.data_param().batch_size(),
          datum.channels(), crop_size, crop_size);
      this->prefetch_data_[top_index]->Reshape(
          this->layer_param_.multidata_param().batch_size(), datum.channels(), 
          crop_size, crop_size);
      this->transformed_data_[top_index]->Reshape(1, datum.channels(), 
          crop_size, crop_size);
    } else {
      top[top_index]->Reshape(
          this->layer_param_.multidata_param().batch_size(), datum.channels(),
          datum.height(), datum.width());
      this->prefetch_data_[top_index]->Reshape(
          this->layer_param_.multidata_param().batch_size(), datum.channels(), 
          datum.height(), datum.width());
      this->transformed_data_[top_index]->Reshape(1, datum.channels(),
          datum.height(), datum.width());
    }
    LOG(INFO) << "output data " << top_index << " size: " 
        << top[top_index]->num() << "," << top[top_index]->channels() << "," 
        << top[top_index]->height() << "," << top[top_index]->width();
  }

  data_transformer_.resize(top_size_);
  for (int top_index = 0; top_index < top_size_; ++top_index) {
    data_transformer_[top_index].reset(
        new DataTransformer<Dtype>(
            this->layer_param_.multidata_param().transform_param(top_index), 
            this->phase_)); 
  }

  // Now, start the prefetch thread. Before calling prefetch, we make a
  // cpu_data call so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  for (int top_index = 0; top_index < top_size_; ++top_index) {
    this->prefetch_data_[top_index]->mutable_cpu_data();
  }
  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void MultidataLayer<Dtype>::CreatePrefetchThread() {
  for (int top_index = 0; 
       top_index < top_size_; 
       ++top_index) {
    this->data_transformer_[top_index]->InitRand();
  }
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void MultidataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void MultidataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  for (int top_index = 0; top_index < top_size_; ++top_index) {
    // Reshape to loaded data.
    top[top_index]->Reshape(this->prefetch_data_[top_index]->num(), 
        this->prefetch_data_[top_index]->channels(), 
        this->prefetch_data_[top_index]->height(), 
        this->prefetch_data_[top_index]->width());
    // Copy the data
    caffe_copy(this->prefetch_data_[top_index]->count(), 
        this->prefetch_data_[top_index]->cpu_data(),
        top[top_index]->mutable_cpu_data());
  }
  DLOG(INFO) << "Prefetch copied";
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  CreatePrefetchThread();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultidataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_[0]->count());
  CHECK(this->transformed_data_[0]->count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.multidata_param().batch_size();
  
  if (batch_size == 1) {
    string binary_data = cursor_->value();
    Datum datum;
    CodedInputStream stream(reinterpret_cast<const uchar*>(binary_data.data()), 
                            binary_data.size());

    for (int top_index = 0; top_index < top_size_; ++top_index) {
      GetNextDatum(&stream, &datum);

      const int crop_size = 
          this->layer_param_.multidata_param().
          transform_param(top_index).crop_size();
      const bool force_color = 
          this->layer_param_.multidata_param().force_encoded_color(top_index);

      if (crop_size != 0) {
        continue;
      }

      if (datum.encoded()) {
        if (force_color) {
          DecodeDatum(&datum, true);
        } else {
          DecodeDatumNative(&datum);
        }
      }
      this->prefetch_data_[top_index]->Reshape(1, datum.channels(),
          datum.height(), datum.width());
      this->transformed_data_[top_index]->Reshape(1, datum.channels(),
          datum.height(), datum.width());
    }
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    string binary_data = cursor_->value();
    Datum datum;
    CodedInputStream stream(reinterpret_cast<const uchar*>(binary_data.data()), 
                            binary_data.size());
    read_time += timer.MicroSeconds();

    for (int top_index = 0; top_index < top_size_; ++top_index) {
      timer.Start();

      GetNextDatum(&stream, &datum);

      const bool force_color = 
          this->layer_param_.multidata_param().force_encoded_color(top_index);

      cv::Mat cv_img;
      if (datum.encoded()) {
        if (force_color) {
          cv_img = DecodeDatumToCVMat(datum, true);
        } else {
          cv_img = DecodeDatumToCVMatNative(datum);
        }
        if (cv_img.channels() != 
            this->transformed_data_[top_index]->channels()) {
          LOG(WARNING) << "Your dataset contains encoded images with mixed "
          << "channel sizes. Consider adding a 'force_color' flag to the "
          << "model definition, or rebuild your dataset using "
          << "convert_imageset.";
        }
      }
      read_time += timer.MicroSeconds();

      timer.Start();
    
      // Apply data transformations (mirror, scale, crop...)
      Dtype* top_data = this->prefetch_data_[top_index]->mutable_cpu_data();
      int offset = this->prefetch_data_[top_index]->offset(item_id);
      this->transformed_data_[top_index]->set_cpu_data(top_data + offset);
      if (datum.encoded()) {
        this->data_transformer_[top_index]->
            Transform(cv_img, this->transformed_data_[top_index].get());
      } else {
        this->data_transformer_[top_index]->
            Transform(datum, this->transformed_data_[top_index].get());
      }
      trans_time += timer.MicroSeconds();
    }
    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MultidataLayer, Forward);
#endif

INSTANTIATE_CLASS(MultidataLayer);
REGISTER_LAYER_CLASS(Multidata);

}  // namespace caffe

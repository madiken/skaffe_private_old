#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void EntropyLossForwardGPU(const int nthreads,
          const Dtype* prob_data, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    loss[index] = -prob_data[index] * log(max(prob_data[index], 
                                              Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* entropy_data = entropy_.mutable_gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  const int nthreads = prob_.count();
  const int channels = bottom[0]->shape(softmax_axis_);

  // NOLINT_NEXT_LINE(whitespace/operators)
  EntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, loss_data);
  // Compute entropies for each pixel.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, loss_data,
      entropy_data);

  Dtype loss;
  caffe_gpu_asum(outer_num_ * inner_num_, entropy_data, &loss);
  loss /= outer_num_;
  top[0]->mutable_cpu_data()[0] = loss;

  // std::cout << "[DEBUG] " << loss << std::endl;

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void EntropyLossBackwardGPU(const int nthreads, 
          const Dtype* prob_data, const Dtype* entropy_data, Dtype* bottom_diff, 
          const int spatial_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const Dtype entropy_value = entropy_data[n];
    bottom_diff[index] = -prob_data[index] * 
                         (entropy_value + log(max(prob_data[index], 
                                                  Dtype(FLT_MIN))));
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* entropy_data = entropy_.gpu_data();
    const int nthreads = prob_.count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    EntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, entropy_data, 
        bottom_diff, inner_num_);

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
  }

  // {
  //   shared_ptr<BlobProto> blobp(new BlobProto());
  //   bottom[0]->ToProto(blobp.get(), true);
  //   WriteProtoToBinaryFile(*blobp, "bottom.binaryproto");
  // }

  // exit(0);
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyLossLayer);

}  // namespace caffe

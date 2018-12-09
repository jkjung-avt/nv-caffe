#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int idx = n * dim + label_value * spatial_dim + s;
    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[idx], min_dtype<Dtype>()));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts,
          Dtype* weighted_counts, const float* class_weights_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int idx = n * dim + label_value * spatial_dim + s;
    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      loss[index] = 0;
      counts[index] = 0;
      weighted_counts[index] = 0;
    } else {
      float w = class_weights_data[label_value];
      loss[index] = -log(max(prob_data[idx], min_dtype<Dtype>())) * w;
      counts[index] = 1;
      weighted_counts[index] = w;
    }
  }
}

template <>
__global__ void SoftmaxLossForwardGPU<half>(const int nthreads,
    const half* prob_data, const half* label, half* loss,
    const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    half* counts) {
  const float minh = __half2float(min_dtype<half>());
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));
    const int idx = n * dim + label_value * spatial_dim + s;
    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      loss[index].setx(0U);
      counts[index].setx(0U);
    } else {
      loss[index] = float2half_clip(- log(max(__half2float(prob_data[idx]), minh)));
      counts[index].setx(0x3c00U);  // set to 1
    }
  }
}

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Ftype* prob_data = prob_->template gpu_data<Ftype>();
  const Ftype* label = bottom[1]->gpu_data<Ftype>();
  const int dim = prob_->count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  Ftype* loss_data = loss_data_.mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Ftype* counts = prob_->template mutable_gpu_diff<Ftype>();
  Ftype valid_weighted_count = -1;
  cudaStream_t stream = Caffe::thread_stream();
  if (tp<Ftype>() == FLOAT16) {
    if (has_class_weights_) {
      LOG(FATAL) << "FLOAT16 version of SoftmaxWithLoss::Forward_gpu() has not been implemented.";
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossForwardGPU<half><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS, 0, stream>>>(nthreads, reinterpret_cast<const half*>(prob_data),
        reinterpret_cast<const half*>(label), reinterpret_cast<half*>(loss_data),
        prob_->count(), dim, inner_num_, has_ignore_label_, ignore_label_,
        reinterpret_cast<half*>(counts));
  } else {
    if (has_class_weights_) {
      TBlob<Ftype> weighted_counts_blob;
      vector<int> wc_shape(1, nthreads);
      weighted_counts_blob.Reshape(wc_shape);
      Ftype* weighted_counts = weighted_counts_blob.mutable_gpu_data();
      const float* class_weights_data = class_weights_.gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardGPU<Ftype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS, 0, stream>>> (nthreads, prob_data, label, loss_data,
          prob_->count(), dim, inner_num_, has_ignore_label_, ignore_label_, counts,
          weighted_counts, class_weights_data);
      caffe_gpu_asum(nthreads, weighted_counts, &valid_weighted_count, 0);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossForwardGPU<Ftype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS, 0, stream>>> (nthreads, prob_data, label, loss_data,
          prob_->count(), dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
 
  float loss;
  caffe_gpu_asum(nthreads, loss_data, &loss, 0);
  int valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid outputs.
  if (has_class_weights_ ||
      (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)) {
    float float_count;
    caffe_gpu_asum(nthreads, counts, &float_count, 0);
    valid_count = int(float_count);
  }
  float normalizer = get_normalizer(normalization_, valid_count);
  if (has_class_weights_) {
    CHECK_GE(valid_weighted_count, valid_count) << \
        "valid_weighted_count should be greater than or equal to valid_count.";
    top[0]->mutable_cpu_data<Ftype>()[0] = (loss / normalizer) /
                                           (valid_weighted_count / valid_count);
  } else {
    top[0]->mutable_cpu_data<Ftype>()[0] = loss / normalizer;
  }
  if (top.size() == 2) {
    top[1]->ShareData(*prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts,
          Dtype label_smoothing) {
  const int channels = dim / spatial_dim;
  const float p1 = 1.F - float(label_smoothing);

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int idx = n * dim + label_value * spatial_dim + s;

    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[idx] -= p1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts,
          Dtype label_smoothing,
          Dtype* weighted_counts, const float* class_weights_data) {
  const int channels = dim / spatial_dim;
  const float p1 = 1.F - float(label_smoothing);

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int idx = n * dim + label_value * spatial_dim + s;

    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
      weighted_counts[index] = 0;
    } else {
      float w = class_weights_data[label_value];
      bottom_diff[idx] -= p1;
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] *= w;
      }
      counts[index] = 1;
      weighted_counts[index] = w;
    }
  }
}

template <>
__global__ void SoftmaxLossBackwardGPU<half>(const int nthreads, const half* top,
    const half* label, half* bottom_diff, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, half* counts, half label_smoothing) {
  const int channels = dim / spatial_dim;
  const float p1 = 1.F - __half2float(label_smoothing);
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));
    const int idx = n * dim + label_value * spatial_dim + s;

    if ((has_ignore_label_ && label_value == ignore_label_) || idx >= num) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s].setx(0U);
      }
      counts[index].setx(0U);
    } else {
      bottom_diff[idx] = float2half_clip(__half2float(bottom_diff[idx]) - p1);
      counts[index].setx(0x3c00U);  // 1.
    }
  }
}


template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    Btype* prob_data = prob_->template mutable_gpu_data<Btype>();
    const Btype* top_data = top[0]->gpu_data<Btype>();
    caffe_gpu_memcpy(prob_->count() * sizeof(Btype), prob_data, bottom_diff);
    const Btype* label = bottom[1]->gpu_data<Btype>();
    const int dim = prob_->count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    Btype* counts = prob_->template mutable_gpu_diff<Btype>();
    Btype valid_weighted_count = -1;
    if (label_smoothing_ > 0) {
      caffe_gpu_add_scalar(bottom[0]->count(),
                           (Btype)( - label_smoothing_ / (dim - 1)), bottom_diff);
    }
    if (has_class_weights_) {
      TBlob<Btype> weighted_counts_blob;
      vector<int> wc_shape(1, nthreads);
      weighted_counts_blob.Reshape(wc_shape);
      Btype* weighted_counts = weighted_counts_blob.mutable_gpu_data();
      const float* class_weights_data = class_weights_.gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(nthreads, top_data, label, bottom_diff,
          bottom[0]->count(), dim, inner_num_, has_ignore_label_, ignore_label_, counts,
          (Btype)label_smoothing_,
          weighted_counts, class_weights_data);
      caffe_gpu_asum(nthreads, weighted_counts, &valid_weighted_count, 0);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SoftmaxLossBackwardGPU<<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(nthreads, top_data, label, bottom_diff,
          bottom[0]->count(), dim, inner_num_, has_ignore_label_, ignore_label_, counts,
          (Btype)label_smoothing_);
    }
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

    int valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (has_class_weights_ ||
	(normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_)) {
      float float_count;
      caffe_gpu_asum(nthreads, counts, &float_count, 0);
      valid_count = int(float_count);
    }
    float normalizer = get_normalizer(normalization_, valid_count);
    float loss_weight;
    if (has_class_weights_) {
      CHECK_GE(valid_weighted_count, valid_count) << \
          "valid_weighted_count should be greater than or equal to valid_count.";
      loss_weight = (float(top[0]->cpu_diff<Btype>()[0]) / normalizer) /
                    (valid_weighted_count / valid_count);
    } else {
      loss_weight = float(top[0]->cpu_diff<Btype>()[0]) / normalizer;
    }
    if (this->parent_net() != NULL) {
      loss_weight *= this->parent_net()->global_grad_scale();
    }
    caffe_gpu_scal<Btype>(prob_->count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SoftmaxWithLossLayer);

}  // namespace caffe

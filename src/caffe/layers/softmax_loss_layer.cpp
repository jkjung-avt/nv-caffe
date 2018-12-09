#include <algorithm>

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif
#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
#ifdef USE_CUDNN
  softmax_layer_.reset(new CuDNNSoftmaxLayer<Ftype, Btype>(softmax_param));
#else
  softmax_layer_.reset(new SoftmaxLayer<Ftype, Btype>(softmax_param));
#endif
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(prob_.get());
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  label_smoothing_ = this->layer_param_.loss_param().label_smoothing();
  has_class_weights_ = false;  // default
}

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::SetClassWeights(
    const vector<float> class_weights) {
  vector<int> cw_shape(1, class_weights.size());
  class_weights_.Reshape(cw_shape);
  float* class_weights_data = class_weights_.mutable_cpu_data<float>();
  for (int i = 0; i < class_weights.size(); ++i) {
    LOG(INFO) << "class_weights[" << i << "] = " << class_weights[i];
    CHECK_GE(class_weights[i], 1.0) << \
        "class_weight value must be greater than or equal to 1.0.";
    class_weights_data[i] = class_weights[i];
  }
  has_class_weights_ = true;
}

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (has_class_weights_) {
    CHECK_EQ(bottom[0]->shape(softmax_axis_), class_weights_.count())
      << "Number of class_weight's ("
      << class_weights_.count()
      << ") should be equal to number of elements in the "
      << "softmax_axis dimension ("
      << bottom[0]->shape(softmax_axis_)
      << ").";
  }
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  loss_data_.ReshapeLike(*bottom[0]);
}

template <typename Ftype, typename Btype>
float SoftmaxWithLossLayer<Ftype, Btype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  float normalizer = 1.F;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = static_cast<float>(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = static_cast<float>(outer_num_ * inner_num_);
      } else {
        normalizer = static_cast<float>(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = static_cast<float>(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = 1.F;
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(1.F, normalizer);
}

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Ftype* prob_data = prob_->template cpu_data<Ftype>();
  const Ftype* label = bottom[1]->cpu_data<Ftype>();
  int dim = prob_->count() / outer_num_;
  int count = 0;
  float weighted_count = 0.0;
  const float* class_weights_data = 0;
  if (has_class_weights_) {
    class_weights_data = class_weights_.cpu_data<float>();
  }
  float loss = 0.F;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_->shape(softmax_axis_));
      if (has_class_weights_) {
        float w = class_weights_data[label_value];
        loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                             min_dtype<Ftype>())) * w;
        weighted_count += w;
      } else {
        loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
            min_dtype<Ftype>()));
      }
      ++count;
    }
  }
  float normalizer = get_normalizer(normalization_, count);
  if (has_class_weights_) {
    CHECK_GE(weighted_count, count) << \
        "weighted_count should be greater than or equal to count.";
    top[0]->mutable_cpu_data<Ftype>()[0] = (loss / normalizer) /
                                           (weighted_count / count);
  } else {
    top[0]->mutable_cpu_data<Ftype>()[0] = loss / normalizer;
  }
  if (top.size() == 2) {
    top[1]->ShareData(*prob_);
  }
}

template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    const Btype* prob_data = prob_->template cpu_data<Btype>();
    caffe_copy(prob_->count(), prob_data, bottom_diff);
    const Btype* label = bottom[1]->cpu_data<Btype>();
    int dim = prob_->count() / outer_num_;
    int count = 0;
    float weighted_count = 0.0;
    const float* class_weights_data = 0;
    if (has_class_weights_) {
      class_weights_data = class_weights_.cpu_data<float>();
    }
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0.F;
          }
        } else {
          if (has_class_weights_) {
            float w = class_weights_data[label_value];
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1.F;
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= w;
            }
            weighted_count += w;
          } else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1.F;
          }
          ++count;
        }
      }
    }
    // Scale gradient
    float normalizer = get_normalizer(normalization_, count);
    Btype loss_weight;
    if (has_class_weights_) {
      CHECK_GE(weighted_count, count) << \
          "weighted_count should be greater than or equal to count.";
      loss_weight = (top[0]->cpu_diff<Btype>()[0] / normalizer) /
                    (weighted_count / count);
    } else {
      loss_weight = top[0]->cpu_diff<Btype>()[0] / normalizer;
    }
    if (this->parent_net() != NULL) {
      const float fp16_global_grad_scale = this->parent_net()->global_grad_scale();
      loss_weight *= fp16_global_grad_scale;
    }
    caffe_scal(prob_->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS_FB(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe

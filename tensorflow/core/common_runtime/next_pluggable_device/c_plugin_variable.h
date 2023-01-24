/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_VARIABLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_VARIABLE_H_

#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_variable.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class CPluginOpKernelContext;

class CPluginVariable : public PluginVariable {
 public:
  ~CPluginVariable() override;
  explicit CPluginVariable(TF_VariableInfo* var_info) : var_info_(var_info) {}

  tsl::Status GetTensor(const Tensor** result_tensor) override;

  tsl::Status GetMutableTensor(Tensor** result_tensor) override;

  friend class CPluginOpKernelContext;

 private:
  tsl::Status GetTensorInternal();

  TF_VariableInfo* var_info_;  // Owned. Cleared by destructor.
  bool tensor_obtained_ = false;
  tensorflow::Tensor tensor_;  // Tensor obtained from variable.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_VARIABLE_H_

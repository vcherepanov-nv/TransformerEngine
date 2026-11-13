/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <xla/ffi/api/c_api.h>

#include <array>
#include <atomic>
#include <cstdint>

namespace {
constexpr int kTargets = 5;
std::array<XLA_FFI_Handler*, kTargets> originals{};
std::array<std::atomic<uint64_t>, kTargets> executions{};
std::array<std::atomic<uint64_t>, kTargets> metadata_queries{};
std::array<std::atomic<uint64_t>, kTargets> errors{};

template <int Index>
XLA_FFI_Error* Execute(XLA_FFI_CallFrame* frame) {
  bool metadata = false;
  for (auto* extension = frame->extension_start; extension; extension = extension->next) {
    metadata |= extension->type == XLA_FFI_Extension_Metadata;
  }
  const bool runtime = !metadata && frame->stage == XLA_FFI_ExecutionStage_EXECUTE;
  auto* error = originals[Index](frame);
  if (metadata) ++metadata_queries[Index];
  if (runtime && !error) ++executions[Index];
  if (error) ++errors[Index];
  return error;
}
std::array<XLA_FFI_Handler*, kTargets> wrappers{Execute<0>, Execute<1>, Execute<2>, Execute<3>,
                                                Execute<4>};
}  // namespace

extern "C" void* Install(int index, void* original) {
  if (index < 0 || index >= kTargets || originals[index]) return nullptr;
  originals[index] = reinterpret_cast<XLA_FFI_Handler*>(original);
  return reinterpret_cast<void*>(wrappers[index]);
}
extern "C" uint64_t Executions(int index) { return executions[index].load(); }
extern "C" uint64_t MetadataQueries(int index) { return metadata_queries[index].load(); }
extern "C" uint64_t Errors(int index) { return errors[index].load(); }

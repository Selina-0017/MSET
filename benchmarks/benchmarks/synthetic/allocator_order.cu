#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK_BOOL(call)                                                  \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

constexpr int FREE_PAGE_COUNT = 4;
constexpr int REQUEST_START = 4;
constexpr int REQUEST_COUNT = 1;
constexpr int OUTPUT_ELEMENTS = 8;

__global__ void allocDecodeKernel(const int *freePages, int *output,
                                  int requestStart) {
  int idx = threadIdx.x;
  if (idx == 0) {
    output[0] = freePages[requestStart];
  }
}

__global__ void safeAllocDecodeKernel(const int *freePages, int *output) {
  int idx = threadIdx.x;
  if (idx == 0) {
    output[0] = freePages[0];
  }
}

void fillFreePages(int *buffer, int n) {
  for (int i = 0; i < n; ++i) {
    buffer[i] = 1000 + i;
  }
}

uint64_t hashOutput(const int *buffer, int n) {
  uint64_t hash = 1469598103934665603ULL;
  for (int i = 0; i < n; ++i) {
    hash ^= static_cast<uint64_t>(buffer[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool hasEnoughFreePages(int freePageCount, int requestStart, int requestCount) {
  return requestStart >= 0 && requestCount >= 0 &&
         requestStart + requestCount <= freePageCount;
}

bool runCase(const char *name) {
  int *hostFreePages =
      static_cast<int *>(std::malloc(sizeof(int) * FREE_PAGE_COUNT));
  int *hostOutput =
      static_cast<int *>(std::malloc(sizeof(int) * OUTPUT_ELEMENTS));
  if (hostFreePages == nullptr || hostOutput == nullptr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(hostFreePages);
    std::free(hostOutput);
    return false;
  }

  fillFreePages(hostFreePages, FREE_PAGE_COUNT);
  std::memset(hostOutput, 0, sizeof(int) * OUTPUT_ELEMENTS);

  int *deviceFreePages = nullptr;
  int *deviceOutput = nullptr;
  CUDA_CHECK_BOOL(cudaMalloc(&deviceFreePages, sizeof(int) * FREE_PAGE_COUNT));
  CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(int) * OUTPUT_ELEMENTS));
  CUDA_CHECK_BOOL(cudaMemcpy(deviceFreePages, hostFreePages,
                             sizeof(int) * FREE_PAGE_COUNT,
                             cudaMemcpyHostToDevice));
  CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(int) * OUTPUT_ELEMENTS));

  bool expectedEarlyReject =
      !hasEnoughFreePages(FREE_PAGE_COUNT, REQUEST_START, REQUEST_COUNT);

  if (strcmp(name, "safe_alloc_decode") == 0) {
    if (expectedEarlyReject) {
      std::printf("case=%s early_reject=true output_hash=%llu sync_status=%s\n",
                  name,
                  static_cast<unsigned long long>(
                      hashOutput(hostOutput, OUTPUT_ELEMENTS)),
                  cudaGetErrorString(cudaSuccess));
      CUDA_CHECK_BOOL(cudaFree(deviceFreePages));
      CUDA_CHECK_BOOL(cudaFree(deviceOutput));
      std::free(hostFreePages);
      std::free(hostOutput);
      return true;
    }
    safeAllocDecodeKernel<<<1, 1>>>(deviceFreePages, deviceOutput);
  } else if (strcmp(name, "late_bounds_check_oob") == 0) {
    allocDecodeKernel<<<1, 1>>>(deviceFreePages, deviceOutput, REQUEST_START);
  } else {
    std::fprintf(stderr, "unknown case: %s\n", name);
    CUDA_CHECK_BOOL(cudaFree(deviceFreePages));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostFreePages);
    std::free(hostOutput);
    return false;
  }

  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
    CUDA_CHECK_BOOL(cudaFree(deviceFreePages));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostFreePages);
    std::free(hostOutput);
    return false;
  }

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
  }

  cudaError_t copyErr =
      cudaMemcpy(hostOutput, deviceOutput, sizeof(int) * OUTPUT_ELEMENTS,
                 cudaMemcpyDeviceToHost);
  if (copyErr != cudaSuccess) {
    std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
  }

  std::printf(
      "case=%s early_reject=false output_hash=%llu sync_status=%s "
      "copy_status=%s\n",
      name,
      static_cast<unsigned long long>(hashOutput(hostOutput, OUTPUT_ELEMENTS)),
      cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

  CUDA_CHECK_BOOL(cudaFree(deviceFreePages));
  CUDA_CHECK_BOOL(cudaFree(deviceOutput));
  std::free(hostFreePages);
  std::free(hostOutput);
  return syncErr == cudaSuccess && copyErr == cudaSuccess;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <case>\n", argv[0]);
    return 1;
  }

  bool ok = runCase(argv[1]);
  return ok ? 0 : 3;
}

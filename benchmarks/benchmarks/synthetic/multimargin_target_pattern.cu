#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK_BOOL(call)                                                   \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err));                    \
            return false;                                                       \
        }                                                                       \
    } while (0)

constexpr int CLASSES = 5;
constexpr int ROWS = 3;

__global__ void safeTargetLookupKernel(const float *input, const int *targets, float *output, int classes) {
    int row = threadIdx.x;
    if (row >= ROWS) {
        return;
    }

    int target = targets[row];
    output[row] = input[row * classes + target];
}

__global__ void negativeTargetLookupKernel(const float *input, const int *targets, float *output, int classes) {
    int row = threadIdx.x;
    if (row >= ROWS) {
        return;
    }

    int target = targets[row];
    const float *rowBase = input + row * classes;
    output[row] = rowBase[target];
}

void fillInput(float *buffer, int count) {
    for (int i = 0; i < count; ++i) {
        buffer[i] = 10.0f + static_cast<float>(i);
    }
}

uint64_t hashOutput(const float *buffer, int n) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &buffer[i], sizeof(uint32_t));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool runCase(const char *name) {
    int hostTargets[ROWS] = {0, 1, 2};
    bool bugCase = false;

    if (strcmp(name, "safe_multimargin_target_read") == 0) {
        bugCase = false;
    } else if (strcmp(name, "negative_multimargin_target_read") == 0) {
        hostTargets[0] = -1;
        hostTargets[1] = -1;
        hostTargets[2] = -1;
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    float hostInput[ROWS * CLASSES];
    fillInput(hostInput, ROWS * CLASSES);
    float hostOutput[ROWS] = {0.0f, 0.0f, 0.0f};

    float *deviceInput = nullptr;
    int *deviceTargets = nullptr;
    float *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceInput, sizeof(float) * ROWS * CLASSES));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceTargets, sizeof(int) * ROWS));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(float) * ROWS));
    CUDA_CHECK_BOOL(
        cudaMemcpy(deviceInput, hostInput, sizeof(float) * ROWS * CLASSES, cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_BOOL(cudaMemcpy(deviceTargets, hostTargets, sizeof(int) * ROWS, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(float) * ROWS));

    if (bugCase) {
        negativeTargetLookupKernel<<<1, ROWS>>>(deviceInput, deviceTargets, deviceOutput, CLASSES);
    } else {
        safeTargetLookupKernel<<<1, ROWS>>>(deviceInput, deviceTargets, deviceOutput, CLASSES);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceTargets));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
    }

    cudaError_t copyErr = cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * ROWS, cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s rows=%d classes=%d target0=%d output_hash=%llu sync_status=%s copy_status=%s\n",
                name, ROWS, CLASSES, hostTargets[0],
                static_cast<unsigned long long>(hashOutput(hostOutput, ROWS)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceInput));
    CUDA_CHECK_BOOL(cudaFree(deviceTargets));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
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

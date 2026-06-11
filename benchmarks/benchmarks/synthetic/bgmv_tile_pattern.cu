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

constexpr int SEQ_LEN = 32768;
constexpr int BLOCK_THREADS = 160;
constexpr int VEC_SIZE = 4;
constexpr int TILE_COLS = BLOCK_THREADS * VEC_SIZE;
constexpr int SAFE_FEAT_IN = TILE_COLS;
constexpr int BUG_FEAT_IN = 512;
constexpr int RANK = 16;

__global__ void safeBgmvTileKernel(const float *input, float *output, int featIn, int row) {
    int lane = threadIdx.x;
    int baseCol = lane * VEC_SIZE;
    int base = row * featIn;
    float accum = 0.0f;

    for (int i = 0; i < VEC_SIZE; ++i) {
        int col = baseCol + i;
        if (col < featIn) {
            accum += input[base + col];
        }
    }

    atomicAdd(&output[lane % RANK], accum);
}

__global__ void bgmvTileOverflowKernel(const float *input, float *output, int featIn, int row) {
    int lane = threadIdx.x;
    int baseCol = lane * VEC_SIZE;
    int base = row * featIn;
    float accum = 0.0f;

    for (int i = 0; i < VEC_SIZE; ++i) {
        accum += input[base + baseCol + i];
    }

    atomicAdd(&output[lane % RANK], accum);
}

void fillInput(float *buffer, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buffer[i] = 0.25f + static_cast<float>(i % 97) * 0.5f;
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
    int featIn = 0;
    bool bugCase = false;

    if (strcmp(name, "safe_bgmv_tile_read") == 0) {
        featIn = SAFE_FEAT_IN;
        bugCase = false;
    } else if (strcmp(name, "bgmv_tile_overflow_read") == 0) {
        featIn = BUG_FEAT_IN;
        bugCase = true;
    } else {
        std::fprintf(stderr, "unknown case: %s\n", name);
        return false;
    }

    const int row = SEQ_LEN - 1;
    const size_t inputCount = static_cast<size_t>(SEQ_LEN) * static_cast<size_t>(featIn);

    float *hostInput = static_cast<float *>(std::malloc(sizeof(float) * inputCount));
    float *hostOutput = static_cast<float *>(std::malloc(sizeof(float) * RANK));
    if (hostInput == nullptr || hostOutput == nullptr) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    fillInput(hostInput, inputCount);
    std::memset(hostOutput, 0, sizeof(float) * RANK);

    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&deviceInput, sizeof(float) * inputCount));
    CUDA_CHECK_BOOL(cudaMalloc(&deviceOutput, sizeof(float) * RANK));
    CUDA_CHECK_BOOL(cudaMemcpy(deviceInput, hostInput, sizeof(float) * inputCount, cudaMemcpyHostToDevice));
    CUDA_CHECK_BOOL(cudaMemset(deviceOutput, 0, sizeof(float) * RANK));

    if (bugCase) {
        bgmvTileOverflowKernel<<<1, BLOCK_THREADS>>>(deviceInput, deviceOutput, featIn, row);
    } else {
        safeBgmvTileKernel<<<1, BLOCK_THREADS>>>(deviceInput, deviceOutput, featIn, row);
    }

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launchErr));
        CUDA_CHECK_BOOL(cudaFree(deviceInput));
        CUDA_CHECK_BOOL(cudaFree(deviceOutput));
        std::free(hostInput);
        std::free(hostOutput);
        return false;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::fprintf(stderr, "sync error: %s\n", cudaGetErrorString(syncErr));
    }

    cudaError_t copyErr =
        cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * RANK, cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::fprintf(stderr, "copy error: %s\n", cudaGetErrorString(copyErr));
    }

    std::printf("case=%s seq_len=%d feat_in=%d tile_cols=%d rank=%d output_hash=%llu "
                "sync_status=%s copy_status=%s\n",
                name, SEQ_LEN, featIn, TILE_COLS, RANK,
                static_cast<unsigned long long>(hashOutput(hostOutput, RANK)),
                cudaGetErrorString(syncErr), cudaGetErrorString(copyErr));

    CUDA_CHECK_BOOL(cudaFree(deviceInput));
    CUDA_CHECK_BOOL(cudaFree(deviceOutput));
    std::free(hostInput);
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

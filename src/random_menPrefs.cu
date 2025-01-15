#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "CudaProgressBar.cuh"
#include "stable_marriage.cuh"

constexpr int threads_per_block = 32;
constexpr int total_blocks = 1 << 18;
constexpr ull total_threads = ull(threads_per_block) * total_blocks;

template <int n, int matchCountSize>
__global__ void countAll2AllStableMarriages(ull offset, ull menPrefArr, ull *d_counts) {
    const ull womenPrefId = ((ull)blockIdx.x * blockDim.x + threadIdx.x + offset) * factorial_v<n>;
    if (womenPrefId >= nrInstances<n>) return;
    int count = countStableMarriage<n>(menPrefArr, id_to_arr<n>(womenPrefId));
    atomicAdd(&d_counts[blockIdx.x * matchCountSize + count - 1], 1);
}

template <int n, int matchCountSize>
__host__ int countStableMarriages(ull menPrefId) {
    copyPermsToDevice<n>();

    thrust::device_vector<ull> d_counts(matchCountSize * total_blocks);

    std::cout << "Calculating n=" << n << " matchCountSize=" << matchCountSize
              << " nrInstances=" << nrInstances<n> << " menPrefId=" << menPrefId << std::endl;
    displayPrefsId<n>(menPrefId);

    constexpr int total_runs = (nrInstances<n> / factorial_v<n> - 1 + total_threads) / total_threads;

    CudaProgressBar<total_runs> progress;
    for (int i = 0; i < total_runs; ++i) {
        countAll2AllStableMarriages<n, matchCountSize><<<total_blocks, threads_per_block>>>(
            i * total_threads, id_to_arr<n>(menPrefId), thrust::raw_pointer_cast(d_counts.data()));
        progress.submited();
    }
    auto time = progress.finish();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << "ms" << std::endl;

    thrust::host_vector<ull> h_counts(d_counts), counts(matchCountSize);
    for (int i = 0; i < matchCountSize * total_blocks; ++i) counts[i % matchCountSize] += h_counts[i];
    for (int i = 0; i < matchCountSize; ++i) std::cout << (i ? "," : "") << counts[i];
    std::cout << '\n';
    int max_count = 0;
    for (int i = 0; i < matchCountSize; ++i)
        if (counts[i] > 0) max_count = i + 1;
    std::cout << "Max count: " << max_count << '\n' << std::endl;
    return max_count;
}

int main(int argc, char **argv) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    // countStableMarriages<1, 1>(distribution<1>(rng));
    // countStableMarriages<2, 2>(distribution<2>(rng));
    // countStableMarriages<3, 3>(distribution<3>(rng));
    // countStableMarriages<4, 10>(distribution<4>(rng));
    int max_count = 0;
    std::vector<ull> max_menPrefIds;
    constexpr int n = 5, matchCountSize = 16;
    constexpr bool show_max = false;
    for (;;) {
        ull menPrefId = argc == 2 ? std::stoull(argv[1]) : distribution<n>(rng) * factorial_v<n - 1>;
        int count = countStableMarriages<n, matchCountSize>(menPrefId);
        if (count > max_count) {
            max_count = count;
            if (show_max) max_menPrefIds = {menPrefId};
        } else if (show_max && count == max_count) {
            max_menPrefIds.push_back(menPrefId);
        }
        std::cerr << "Max count: " << max_count;
        if (show_max) {
            std::cerr << " menPrefIds: [";
            for (int i = 0; i < max_menPrefIds.size(); ++i) std::cerr << (i ? "," : "") << max_menPrefIds[i];
            std::cerr << "]\n" << std::endl;
        } else {
            std::cerr << std::endl;
        }
    }
}
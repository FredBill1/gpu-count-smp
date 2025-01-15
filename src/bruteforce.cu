#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <iostream>

#include "stable_marriage.cuh"

constexpr int threads_per_block = 32;
constexpr int total_blocks = 1 << 18;
constexpr ull total_threads = ull(threads_per_block) * total_blocks;

template <int n, int matchCountSize>
__global__ void countAll2AllStableMarriages(ull *d_counts) {
    const ull womenPrefId = (blockIdx.x * threads_per_block + threadIdx.x) * factorial_v<n - 1>;
    if (womenPrefId >= nrInstances<n>) return;
    const ull womenPrefArr = id_to_arr<n>(womenPrefId);
    for (ull menPrefId = 0; menPrefId < nrInstances<n>; menPrefId += factorial_v<n>) {
        int count = countStableMarriage<n>(id_to_arr<n>(menPrefId), womenPrefArr);
        atomicAdd(&d_counts[blockIdx.x * matchCountSize + count - 1], 1);
    }
}

template <int n, int matchCountSize>
__host__ int countStableMarriages() {
    copyPermsToDevice<n>();

    thrust::device_vector<ull> d_counts(matchCountSize * total_blocks);

    std::cout << "Calculating n=" << n << " matchCountSize=" << matchCountSize
              << " (n!)^n=" << nrInstances<n> << std::endl;

    constexpr int total_runs = (nrInstances<n> / factorial_v<n - 1> - 1 + total_threads) / total_threads;

    auto start = std::chrono::high_resolution_clock::now();
    countAll2AllStableMarriages<n, matchCountSize>
        <<<total_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_counts.data()));
    cudaDeviceSynchronize();
    auto time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << "ms" << std::endl;

    thrust::host_vector<ull> h_counts(d_counts), counts(matchCountSize);
    for (int i = 0; i < matchCountSize * total_blocks; ++i) counts[i % matchCountSize] += h_counts[i];
    ull sum = 0;
    for (int i = 0; i < matchCountSize; ++i) {
        ull count = counts[i] * (factorial_v<n> * factorial_v<n - 1>);
        std::cout << (i ? "," : "") << count;
        sum += count;
    }
    std::cout << '\n';
    int max_count = 0;
    for (int i = 0; i < matchCountSize; ++i)
        if (counts[i] > 0) max_count = i + 1;
    std::cout << "Max count: " << max_count;
    std::cout << "\nSum: " << sum << (sum == power_v<factorial_v<n>, 2 * n> ? " ==" : " !=") << " (n!)^{2n}\n"
              << std::endl;
    return max_count;
}

int main(int argc, char **argv) {
    countStableMarriages<1, 1>();
    countStableMarriages<2, 2>();
    countStableMarriages<3, 3>();
    countStableMarriages<4, 10>();
}
#include "CudaProgressBar.cuh"
#include "stable_marriage.cuh"

constexpr int threads_per_block = 32;
constexpr int total_blocks = 1 << 18;
constexpr ull total_threads = (ull)threads_per_block * total_blocks;

template <int n>
__global__ void getWomenPrefIdWithMaxCount(ull offset, ull menPrefArr, int targetCount) {
    const ull womenPrefId = ((ull)blockIdx.x * blockDim.x + threadIdx.x + offset) * factorial_v<n>;
    if (womenPrefId >= nrInstances<n>) return;
    int count = countStableMarriage<n>(menPrefArr, id_to_arr<n>(womenPrefId));
    if (count == targetCount) printf("\n%llu\n", womenPrefId);
}

template <int n>
__host__ void getWomenPrefIdWithMaxCount(ull menPrefId, int targetCount) {
    copyPermsToDevice<n>();

    ull menPrefArr = id_to_arr<n>(menPrefId);
    constexpr ull total_runs = (nrInstances<n> / factorial_v<n> + total_threads - 1) / total_threads;
    CudaProgressBar<total_runs> progress;
    for (int i = 0; i < total_runs; ++i) {
        getWomenPrefIdWithMaxCount<n><<<total_blocks, threads_per_block>>>(i * total_threads, menPrefArr, targetCount);
        progress.submited();
    }
    progress.finish();
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cout << "Usage: " << argv[0] << " <menPrefId> <targetCount> [<womenPrefId>]" << std::endl;
        return 1;
    }
    constexpr int n = 5;
    ull menPrefId = std::stoull(argv[1]);
    int targetCount = std::stoi(argv[2]);
    std::cout << "menPrefId: " << menPrefId << " targetCount: " << targetCount << '\n';
    displayPrefsId<n>(menPrefId);
    std::cout << std::endl;
    if (argc == 3) {
        getWomenPrefIdWithMaxCount<n>(menPrefId, targetCount);
    } else {
        ull womenPrefId = std::stoull(argv[3]);
        std::cout << "womenPrefId: " << womenPrefId << '\n';
        displayPrefsId<n>(womenPrefId);
        std::cout << std::endl;
        int count = countStableMarriage<n>(id_to_arr<n>(menPrefId), id_to_arr<n>(womenPrefId));
        std::cout << "count: " << count << (count == targetCount ? " ==" : " !=") << " targetCount" << std::endl;
        return count == targetCount ? 0 : 1;
    }
}
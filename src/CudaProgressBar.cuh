#ifndef _CUDA_PROGRESS_BAR_CUH
#define _CUDA_PROGRESS_BAR_CUH

#include <chrono>
#include <limits>
#include <semaphore>
#include <thread>
#include <vector>

#include "stable_marriage.cuh"
#include "tqdm.h"

template <ull max_total_runs = std::numeric_limits<int>::max()>
class CudaProgressBar {
    bool is_terminal;
    tqdm bar;
    std::counting_semaphore<max_total_runs> sem;
    std::jthread print_thread;
    std::vector<cudaEvent_t> events;
    std::chrono::high_resolution_clock::time_point start_time;
    size_t total_runs_, current_ = 0;

   public:
    CudaProgressBar(size_t total_runs = max_total_runs) : total_runs_(total_runs), sem(0) {
        is_terminal = isatty(fileno(stderr));
        if (is_terminal) {
            events.resize(total_runs);
            print_thread = std::jthread(([this] {
                tqdm bar(stderr);
                for (int i = 0; i < total_runs_; ++i) {
                    bar.progress(i, total_runs_);
                    sem.acquire();
                    cudaEventSynchronize(events[i]);
                }
                bar.finish();
            }));
        }
        start_time = std::chrono::high_resolution_clock::now();
    }
    void submited() {
        if (!is_terminal) return;
        cudaEventCreateWithFlags(&events[current_], cudaEventDisableTiming);
        cudaEventRecord(events[current_]);
        sem.release();
        ++current_;
    }
    std::chrono::high_resolution_clock::duration finish() {
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        if (is_terminal) print_thread.join();
        return end_time - start_time;
    }
};

#endif  // _CUDA_PROGRESS_BAR_CUH

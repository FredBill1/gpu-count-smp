#ifndef _STABLE_MARRIAGE_CUH
#define _STABLE_MARRIAGE_CUH

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <random>

#ifdef _MSC_VER
#pragma warning(disable : 4068)
#endif

using ull = unsigned long long;

__host__ __device__ constexpr ull factorial(ull n) {
    ull result = 1;
    for (ull i = 2; i <= n; ++i) result *= i;
    return result;
}
template <ull n>
constexpr ull factorial_v = factorial(n);

__host__ __device__ constexpr ull power(ull base, ull exp) {
    ull result = 1, p = base, e = exp;
    while (e) {
        if (e & 1) result *= p;
        p *= p;
        e >>= 1;
    }
    return result;
}
template <ull base, ull exp>
constexpr ull power_v = power(base, exp);

template <int n>
__host__ constexpr std::array<int, factorial_v<n> * n * 2> permutations() {
    std::array<int, factorial_v<n> * n * 2> result{};
    std::array<int, n> perm{}, revPerm{};
    for (int i = 0; i < n; ++i) perm[i] = i;
    size_t idx = 0;
    do {
        for (int i = 0; i < n; ++i) revPerm[perm[i]] = i;
        for (int x : perm) result[idx++] = x;
        for (int x : revPerm) result[idx++] = x;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return result;
}
template <int n>
constexpr ull nrInstances = power_v<factorial_v<n>, n>;
template <int n>
inline auto h_perms = permutations<n>();
template <int n>
__constant__ int d_perms[n * factorial_v<n> * 2];
template <int n>
__host__ inline void copyPermsToDevice() {
    cudaMemcpyToSymbol(d_perms<n>, h_perms<n>.data(), sizeof(*h_perms<n>.data()) * h_perms<n>.size());
}
template <int n>
inline std::uniform_int_distribution<ull> distribution(0, nrInstances<n> / factorial_v<n - 1> - 1);

template <int n>
__host__ __device__ constexpr int bit_width() {
    int res = 0, x = n;
    while (x) x >>= 1, ++res;
    return res;
}

template <int n>
__host__ __device__ __forceinline__ ull id_to_arr(ull id) {
    constexpr int width = bit_width<factorial_v<n>>();
    static_assert(std::numeric_limits<ull>::digits >= n * width, "ull is too small");
    ull res = 0;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        res |= (id % factorial_v<n>) << (i * width);
        id /= factorial_v<n>;
    }
    return res;
}

template <int n>
__host__ __device__ __forceinline__ ull arr_to_id(ull arr) {
    constexpr int width = bit_width<factorial_v<n>>();
    constexpr ull mask = (1ull << width) - 1;
    static_assert(std::numeric_limits<ull>::digits >= n * width, "ull is too small");
    ull res = 0;
#pragma unroll
    for (int i = n - 1; i >= 0; --i) {
        res *= factorial_v<n>;
        res += (arr >> (width * i)) & mask;
    }
    return res;
}

template <int n>
__host__ inline void displayPrefsArr(ull prefArr) {
    constexpr int width = bit_width<factorial_v<n>>();
    constexpr unsigned int mask = (1u << width) - 1;
    constexpr int *perms = h_perms<n>.data();
    for (int man = 0; man < n; ++man) {
        const int *prefs = perms + ((prefArr >> (man * width)) & mask) * (n * 2);
        for (int i = 0; i < n; ++i) {
            if (i) std::cout << ',';
            std::cout << prefs[i];
        }
        std::cout << '\n';
    }
    std::cout << std::flush;
}

template <int n>
inline void displayPrefsId(ull prefId) {
    displayPrefsArr<n>(id_to_arr<n>(prefId));
}

template <int n>
__host__ __device__ __forceinline__ bool isStableMarriage(ull menPrefArr, ull womenPrefArr, int menMatchesId) {
    constexpr int width = bit_width<factorial_v<n>>();
    constexpr unsigned int mask = (1u << width) - 1;
#ifdef __CUDA_ARCH__
    constexpr int *perms = d_perms<n>;
#else
    constexpr int *perms = h_perms<n>.data();
#endif
    const int *menMatches = perms + menMatchesId * (n * 2);
    const int *womenMatches = perms + menMatchesId * (n * 2) + n;
#pragma unroll
    for (int man = 0; man < n; ++man) {
        int manMatch = menMatches[man];
        const int *manPrefs = perms + ((menPrefArr >> (man * width)) & mask) * (n * 2);
#pragma unroll
        for (int woman = 0; woman < n; ++woman) {
            if (woman == manMatch) continue;
            int womanMatch = womenMatches[woman];
            const int *womanPrefs = perms + ((womenPrefArr >> (woman * width)) & mask) * (n * 2);
            if (manPrefs[woman] < manPrefs[manMatch] && womanPrefs[man] < womanPrefs[womanMatch]) return false;
        }
    }
    return true;
}

template <int n>
__host__ __device__ __forceinline__ int countStableMarriage(ull menPrefArr, ull womenPrefArr) {
    int count = 0;
    for (int menMatchesId = 0; menMatchesId < factorial_v<n>; ++menMatchesId)
        if (isStableMarriage<n>(menPrefArr, womenPrefArr, menMatchesId)) ++count;
    return count;
}

#endif  // _STABLE_MARRIAGE_CUH

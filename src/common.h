#pragma once

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <source_location>
#include <vector>
#include <list>
#include <stack>
#include <map>
#include <unordered_map>
#include <set>
#include <any>
#include <variant>
#include <optional>
#include <chrono>
#include <functional>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <spdlog/spdlog.h>

class CUDAError : public std::runtime_error {
public:
    CUDAError(cudaError_t errorCode, std::source_location location) 
        : std::runtime_error(format(errorCode, location)), errorCode(errorCode), location(location) {}

public:
    const cudaError_t errorCode;
    const std::source_location location;

private:
    static std::string format(cudaError_t errorCode, std::source_location location) {
        return spdlog::fmt_lib::format("CUDA error: {} (at {}:{})", 
            cudaGetErrorString(errorCode), location.file_name(), location.line());
    }
};

inline cudaError_t checkCUDA(cudaError_t retValue, const std::source_location location = std::source_location::current()) {
    if (retValue != cudaSuccess) {
        throw CUDAError(retValue, location);
    }
    return retValue;
}

inline cublasStatus_t checkCUBLAS(cublasStatus_t retValue, const std::source_location location = std::source_location::current()) {
    if (retValue != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(spdlog::fmt_lib::format("CUBLAS error: {} (at {}:{})", 
            cublasGetStatusString(retValue), location.file_name(), location.line()));
    }
    return retValue;
}

inline thread_local std::stack<cudaStream_t> stackCUDAStreams;

inline cudaStream_t getCurrentCUDAStream() {
    if (stackCUDAStreams.empty()) {
        return 0;
    }
    return stackCUDAStreams.top();
}

inline cudaDeviceProp *getCurrentDeviceProperties() {
    static thread_local cudaDeviceProp prop;
    static thread_local bool propAvailable = false;
    if (!propAvailable) {
        int device;
        checkCUDA(cudaGetDevice(&device));
        checkCUDA(cudaGetDeviceProperties(&prop, device));
        propAvailable = true;
    }
    return &prop;
}

template<typename T>
constexpr T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}

template<typename T>
constexpr int log2Up(T value) {
   if (value <= 0)
       return 0;
   if (value == 1)
       return 0;
   return log2Up((value + 1) / 2) + 1;
}

struct CUBLASWrapper {
    cublasHandle_t handle = nullptr;

    CUBLASWrapper() {
        checkCUBLAS(cublasCreate(&handle));
    }
    CUBLASWrapper(CUBLASWrapper &&) = delete;
    CUBLASWrapper(const CUBLASWrapper &&) = delete;
    ~CUBLASWrapper() {
        if (handle) {
            checkCUBLAS(cublasDestroy(handle));
        }
    }
};

inline std::shared_ptr<CUBLASWrapper> getCUBLAS() {
    static thread_local std::weak_ptr<CUBLASWrapper> inst;
    std::shared_ptr<CUBLASWrapper> result = inst.lock();
    if (result) {
        return result;
    }
    result = std::make_shared<CUBLASWrapper>();
    inst = result;
    return result;
}
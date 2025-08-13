#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Custom exception hierarchy for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& message, cudaError_t error_code, 
                  const char* file, int line)
        : std::runtime_error(format_message(message, error_code, file, line))
        , error_code_(error_code)
    {}

    cudaError_t error_code() const noexcept { return error_code_; }

private:
    cudaError_t error_code_;
    
    static std::string format_message(const std::string& message, 
                                      cudaError_t error_code,
                                      const char* file, int line) {
        return message + " (CUDA error: " + cudaGetErrorString(error_code) + 
               " at " + file + ":" + std::to_string(line) + ")";
    }
};

// Improved error checking macro that throws exceptions
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw CudaException(#call " failed", error, __FILE__, __LINE__); \
    } \
} while(0)

// RAII wrapper for CUDA device memory
template<typename T>
class CudaDeviceMemory {
private:
    T* ptr_;
    size_t size_;
    size_t count_;

public:
    // Constructor allocates memory
    explicit CudaDeviceMemory(size_t count) 
        : ptr_(nullptr), size_(count * sizeof(T)), count_(count) {
        if (count == 0) {
            throw std::invalid_argument("Cannot allocate zero elements");
        }
        CUDA_CHECK(cudaMalloc(&ptr_, size_));
    }

    // Constructor with initialization from host data
    CudaDeviceMemory(const T* host_data, size_t count)
        : CudaDeviceMemory(count) {
        copyFromHost(host_data);
    }

    // Destructor frees memory
    ~CudaDeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Delete copy constructor and assignment operator
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    // Move constructor
    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.count_ = 0;
    }

    // Move assignment operator
    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.count_ = 0;
        }
        return *this;
    }

    // Access methods
    T* get() const noexcept { return ptr_; }
    size_t size_bytes() const noexcept { return size_; }
    size_t count() const noexcept { return count_; }

    // Memory operations
    void copyFromHost(const T* host_data) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, size_, cudaMemcpyHostToDevice));
    }

    void copyToHost(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, size_, cudaMemcpyDeviceToHost));
    }

    void fill(const T& value) {
        // For simple types, we can use cudaMemset for certain values
        if constexpr (std::is_same_v<T, float>) {
            if (value == 0.0f) {
                CUDA_CHECK(cudaMemset(ptr_, 0, size_));
                return;
            }
        }
        
        // For other cases, use a kernel or thrust
        throw std::runtime_error("Fill operation not implemented for this type/value");
    }

    void clear() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_));
    }

    // Synchronization
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Validity check
    bool valid() const noexcept { return ptr_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }
};

// RAII wrapper for CUDA pitched memory (2D arrays)
template<typename T>
class CudaPitchedMemory {
private:
    T* ptr_;
    size_t pitch_;
    size_t width_;
    size_t height_;

public:
    CudaPitchedMemory(size_t width, size_t height) 
        : ptr_(nullptr), pitch_(0), width_(width), height_(height) {
        if (width == 0 || height == 0) {
            throw std::invalid_argument("Cannot allocate zero-sized 2D array");
        }
        CUDA_CHECK(cudaMallocPitch(&ptr_, &pitch_, width * sizeof(T), height));
    }

    ~CudaPitchedMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Delete copy, enable move
    CudaPitchedMemory(const CudaPitchedMemory&) = delete;
    CudaPitchedMemory& operator=(const CudaPitchedMemory&) = delete;

    CudaPitchedMemory(CudaPitchedMemory&& other) noexcept
        : ptr_(other.ptr_), pitch_(other.pitch_), 
          width_(other.width_), height_(other.height_) {
        other.ptr_ = nullptr;
        other.pitch_ = 0;
        other.width_ = 0;
        other.height_ = 0;
    }

    CudaPitchedMemory& operator=(CudaPitchedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            pitch_ = other.pitch_;
            width_ = other.width_;
            height_ = other.height_;
            other.ptr_ = nullptr;
            other.pitch_ = 0;
            other.width_ = 0;
            other.height_ = 0;
        }
        return *this;
    }

    // Access methods
    T* get() const noexcept { return ptr_; }
    size_t pitch() const noexcept { return pitch_; }
    size_t width() const noexcept { return width_; }
    size_t height() const noexcept { return height_; }

    // Row access helper
    T* row(size_t y) const {
        return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + y * pitch_);
    }

    // Memory operations for 2D arrays
    void copyFromHost(const T* host_data, size_t host_pitch) {
        CUDA_CHECK(cudaMemcpy2D(ptr_, pitch_, host_data, host_pitch, 
                               width_ * sizeof(T), height_, cudaMemcpyHostToDevice));
    }

    void copyToHost(T* host_data, size_t host_pitch) const {
        CUDA_CHECK(cudaMemcpy2D(host_data, host_pitch, ptr_, pitch_, 
                               width_ * sizeof(T), height_, cudaMemcpyDeviceToHost));
    }

    void clear() {
        CUDA_CHECK(cudaMemset2D(ptr_, pitch_, 0, width_ * sizeof(T), height_));
    }

    bool valid() const noexcept { return ptr_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }
};

// RAII wrapper for CUDA events
class CudaEvent {
private:
    cudaEvent_t event_;

public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    // Delete copy, enable move
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    cudaEvent_t get() const noexcept { return event_; }

    void record() {
        CUDA_CHECK(cudaEventRecord(event_));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_time(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), event_));
        return ms;
    }
};

// RAII wrapper for CUDA streams
class CudaStream {
private:
    cudaStream_t stream_;

public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Delete copy, enable move
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};

// Utility function to query device properties safely
inline cudaDeviceProp getCudaDeviceProperties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

// Utility function to get device count safely
inline int getCudaDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}
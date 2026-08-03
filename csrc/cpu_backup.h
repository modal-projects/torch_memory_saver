#pragma once

#include "utils.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>

enum class CpuBackupKind : uint8_t {
    MMAP = 0,
    PINNED = 1,
};

#if defined(USE_ROCM)
// ROCm stays pinned-only until mmap reclaim is validated on AMD.
constexpr CpuBackupKind kDefaultCpuBackupKind = CpuBackupKind::PINNED;
#else
constexpr CpuBackupKind kDefaultCpuBackupKind = CpuBackupKind::MMAP;
#endif

struct CpuBackupSlot {
    void* data = nullptr;
    size_t size = 0;
};

// Shared by CUDA (core.cpp) and ROCm (hardware_amd_support.cpp) pause/resume/free.
inline void cpu_backup_release(CpuBackupKind kind, CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        return;
    }
    switch (kind) {
        case CpuBackupKind::MMAP:
            SIMPLE_CHECK(munmap(slot.data, slot.size) == 0,
                         "munmap cpu_backup failed errno=" << errno << " " << strerror(errno));
            break;
        case CpuBackupKind::PINNED:
            CUDA_ERROR_CHECK(cudaFreeHost(slot.data));
            break;
        default:
            SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(kind));
    }
    slot.data = nullptr;
    slot.size = 0;
}

inline void cpu_backup_offload(
    CpuBackupKind kind, void* gpu_ptr, size_t size, CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        switch (kind) {
            case CpuBackupKind::MMAP: {
                void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                SIMPLE_CHECK(p != MAP_FAILED,
                             "mmap cpu_backup failed errno=" << errno << " " << strerror(errno));
                slot.data = p;
                break;
            }
            case CpuBackupKind::PINNED:
                CUDA_ERROR_CHECK(cudaMallocHost(&slot.data, size));
                SIMPLE_CHECK(slot.data != nullptr, "cudaMallocHost cpu_backup returned nullptr");
                break;
            default:
                SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(kind));
        }
        slot.size = size;
    }
    SIMPLE_CHECK(slot.data != nullptr && slot.size == size, "cpu_backup slot size mismatch");
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(slot.data, gpu_ptr, size, cudaMemcpyDeviceToHost));
}

inline void cpu_backup_onload(void* gpu_ptr, size_t size, const CpuBackupSlot& slot) {
    SIMPLE_CHECK(slot.data != nullptr, "cpu_backup missing on resume");
    SIMPLE_CHECK(slot.size == size, "cpu_backup slot size mismatch");
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ptr, slot.data, size, cudaMemcpyHostToDevice));
}

// Shared by tms_set_cpu_backup_backend and TMS_INIT_CPU_BACKUP_BACKEND.
inline CpuBackupKind parse_cpu_backup_kind(const char* value) {
    SIMPLE_CHECK(value != nullptr, "cpu_backup_backend value should not be null");
    std::string s(value);
    if (s == "mmap") {
#if defined(USE_ROCM)
        std::cerr << "[torch_memory_saver.cpp] cpu_backup_backend=mmap is not supported on ROCm"
                  << std::endl;
        exit(1);
#else
        return CpuBackupKind::MMAP;
#endif
    }
    if (s == "pinned") {
        return CpuBackupKind::PINNED;
    }
    std::cerr << "[torch_memory_saver.cpp] cpu_backup_backend must be mmap or pinned"
              << " value=" << s << std::endl;
    exit(1);
}

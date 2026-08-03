#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <optional>
#include "macro.h"
#include "cpu_backup.h"

// ----------------------------------------------- threadlocal configs --------------------------------------------------

class ThreadLocalConfig {
public:
    std::string current_tag_ = "default";

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE");
        }
        return is_interesting_region_.value();
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        if (!enable_cpu_backup_.has_value()) {
            enable_cpu_backup_ = get_bool_env_var("TMS_INIT_ENABLE_CPU_BACKUP");
        }
        return enable_cpu_backup_.value();
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

    // Cached kind, or platform default. Does not parse env (safe for TLS get/restore).
    CpuBackupKind cpu_backup_kind() {
        return cpu_backup_kind_.value_or(kDefaultCpuBackupKind);
    }

    void set_cpu_backup_kind(CpuBackupKind value) {
        cpu_backup_kind_ = value;
    }

    // Parse TMS_INIT_CPU_BACKUP_BACKEND once. Only call when enable_cpu_backup is true
    // so a bad unused env cannot exit(1) on backup-off mallocs.
    void init_cpu_backup_kind_from_env() {
        if (cpu_backup_kind_.has_value()) {
            return;
        }
        const char* env = std::getenv("TMS_INIT_CPU_BACKUP_BACKEND");
        // Empty string is treated as unset (same as Python).
        if (env == nullptr || env[0] == '\0') {
            cpu_backup_kind_ = kDefaultCpuBackupKind;
        } else {
            cpu_backup_kind_ = parse_cpu_backup_kind(env);
        }
    }

    bool enable_disk_backup() {
        if (!enable_disk_backup_.has_value()) {
            enable_disk_backup_ = get_bool_env_var("TMS_INIT_ENABLE_DISK_BACKUP");
        }
        return enable_disk_backup_.value();
    }

    void set_enable_disk_backup(bool value) {
        enable_disk_backup_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
    std::optional<CpuBackupKind> cpu_backup_kind_;
    std::optional<bool> enable_disk_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        const bool enable_cpu_backup = thread_local_config.enable_cpu_backup();
        if (enable_cpu_backup) {
            thread_local_config.init_cpu_backup_kind_from_env();
        }
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_,
            enable_cpu_backup, thread_local_config.cpu_backup_kind(),
            thread_local_config.enable_disk_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    return TorchMemorySaver::instance().free(ptr);
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    const bool enable_cpu_backup = thread_local_config.enable_cpu_backup();
    if (enable_cpu_backup) {
        thread_local_config.init_cpu_backup_kind_from_env();
    }
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_,
        enable_cpu_backup, thread_local_config.cpu_backup_kind(),
        thread_local_config.enable_disk_backup()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().free(ptr));
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.set_interesting_region(is_interesting_region);
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region();
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

const char* tms_get_current_tag() {
    return thread_local_config.current_tag_.c_str();
}

bool tms_get_enable_cpu_backup() {
    return thread_local_config.enable_cpu_backup();
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.set_enable_cpu_backup(enable_cpu_backup);
}

const char* tms_get_cpu_backup_backend() {
    return thread_local_config.cpu_backup_kind() == CpuBackupKind::MMAP ? "mmap" : "pinned";
}

void tms_set_cpu_backup_backend(const char* backend) {
    thread_local_config.set_cpu_backup_kind(parse_cpu_backup_kind(backend));
}

bool tms_get_enable_disk_backup() {
    return thread_local_config.enable_disk_backup();
}

void tms_set_enable_disk_backup(bool enable_disk_backup) {
    thread_local_config.set_enable_disk_backup(enable_disk_backup);
}

void tms_set_disk_backup_dir(const char* dir) {
    SIMPLE_CHECK(dir != nullptr, "disk backup dir should not be null");
    TorchMemorySaver::instance().set_disk_backup_dir(std::string(dir));
}

void set_memory_margin_bytes(uint64_t value) {
    TorchMemorySaver::instance().set_memory_margin_bytes(value);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}
}

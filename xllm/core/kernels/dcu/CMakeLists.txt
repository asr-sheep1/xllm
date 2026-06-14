include(cc_library)
include(utils)

set(DCU_HEADER_FILES
  dcu_ops_api.h
)

set(CUDA_KERNEL_DIR ${CMAKE_SOURCE_DIR}/xllm/core/kernels/cuda)

file(GLOB DCU_CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS
  ${CUDA_KERNEL_DIR}/*.cu
  ${CUDA_KERNEL_DIR}/moe/*.cu
)

file(GLOB DCU_CUDA_KERNEL_HEADERS CONFIGURE_DEPENDS
  ${CUDA_KERNEL_DIR}/*.cuh
  ${CUDA_KERNEL_DIR}/moe/*.cuh
)

list(FILTER DCU_CUDA_KERNEL_SOURCES EXCLUDE REGEX ".*/.*_test\\.cu$")

set(DCU_HIPIFY_EXCLUDED_CUDA_SOURCES
  ${CUDA_KERNEL_DIR}/rec_beam_search.cu
#  ${CUDA_KERNEL_DIR}/fp8_quant.cu
)

set(DCU_HIPIFY_EXCLUDED_CUDA_HEADERS
  # ${CUDA_KERNEL_DIR}/fp8_quant_utils.cuh
)

foreach(excluded IN LISTS DCU_HIPIFY_EXCLUDED_CUDA_SOURCES)
  list(REMOVE_ITEM DCU_CUDA_KERNEL_SOURCES ${excluded})
endforeach()

foreach(excluded IN LISTS DCU_HIPIFY_EXCLUDED_CUDA_HEADERS)
  list(REMOVE_ITEM DCU_CUDA_KERNEL_HEADERS ${excluded})
endforeach()

list(REMOVE_DUPLICATES DCU_CUDA_KERNEL_SOURCES)
list(REMOVE_DUPLICATES DCU_CUDA_KERNEL_HEADERS)

set(DCU_LOCAL_HIP_SOURCES
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/matmul.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/scaled_quantize.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/scaled_matmul.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/build_block_table_from_paged_kv.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/random_sample.hip

  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/attention_runner.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/piecewise_graphs.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/global_capture_instance.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/flash_mla_adapter.cpp
)

# ---------------------------------------------------------------------------
# flash-mla native extension (deepseek-ai/FlashMLA).
#
# Imported as a thin shared library so the DCU MLA decode adapter can call its
# exported C++ symbols directly (see kernels/dcu/flash_mla_adapter.cpp). The
# DCU image is expected to ship flash_mla; fail hard if it is missing rather
# than silently disabling MLA.
# ---------------------------------------------------------------------------
if(NOT TARGET flash_mla)
  if(NOT FLASH_MLA_LIB_DIR)
    set(FLASH_MLA_LIB_DIR "/usr/local/lib/python3.10/dist-packages/flash_mla")
  endif()
  set(FLASH_MLA_LIB_DIR
      "${FLASH_MLA_LIB_DIR}"
      CACHE PATH "Directory containing the flash_mla cuda.*.so native extension")

  file(GLOB _flash_mla_so_paths "${FLASH_MLA_LIB_DIR}/cuda.*.so")
  if(_flash_mla_so_paths)
    list(GET _flash_mla_so_paths 0 _flash_mla_so)
    add_library(flash_mla SHARED IMPORTED)
    set_target_properties(flash_mla PROPERTIES IMPORTED_LOCATION "${_flash_mla_so}")
    set(_flash_mla_rpath "${FLASH_MLA_LIB_DIR}")
    message(STATUS "Found flash_mla: ${_flash_mla_so}")
  else()
    message(FATAL_ERROR
            "flash_mla cuda.*.so not found under ${FLASH_MLA_LIB_DIR}. "
            "Set -DFLASH_MLA_LIB_DIR=<dir> to the flash_mla install dir.")
  endif()
endif()

hipify_sources_target(
  DCU_GENERATED_SRCS
  DCU_HIPIFY_TARGET
  dcu_kernels
  "${DCU_CUDA_KERNEL_SOURCES}"
  EXTRA_HIPIFY_FILES
    ${DCU_CUDA_KERNEL_HEADERS}
)

set(DCU_SOURCE_FILES
  ${DCU_GENERATED_SRCS}
  ${DCU_LOCAL_HIP_SOURCES}
)

cc_library(
  NAME
    dcu_kernels
  HDRS
    ${DCU_HEADER_FILES}
  SRCS
    ${DCU_SOURCE_FILES}
  DEPS
    hip::host
    hipblaslt
    torch
    glog::glog
    :platform
    flash_mla
)

add_dependencies(dcu_kernels ${DCU_HIPIFY_TARGET})

# flash_mla ships outside the standard loader path; keep its dir on the rpath so
# the final xllm binary can resolve cuda.*.so at runtime.
if(DEFINED _flash_mla_rpath)
  set_property(TARGET dcu_kernels APPEND PROPERTY BUILD_RPATH "${_flash_mla_rpath}")
endif()

target_include_directories(dcu_kernels
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

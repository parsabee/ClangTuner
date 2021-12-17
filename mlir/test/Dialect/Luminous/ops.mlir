// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {luminous.container_module} {
  // CHECK-LABEL: luminous.module @kernels
  luminous.module @kernels {
  }
}

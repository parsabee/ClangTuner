// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {luminous.container_module} {
  luminous.module @kernels {
    luminous.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>){
      return
    }
  }

  func @foo() {
    // CHECK: luminous.dispatch @kernels::@kernel_1 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) args(%{{.*}} : f32, %{{.*}} : memref<?xf32, 1>)
    %t1 = luminous.dispatch @kernels::@kernel_1 (%0 : f32, %1 : memref<?xf32, 1>)
    return
  }
}

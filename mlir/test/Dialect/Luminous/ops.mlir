// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {luminous.container_module} {
  luminous.module @kernels {
    // CHECK-LABEL: luminous.module @kernel
  }

  func @function() {
    // CHECK-LABEL: func @function()
    // CHECK: {{.*}} = luminous.wait async
    %t0 = luminous.wait async
    // CHECK: {{.*}} = luminous.wait async
    %t1 = luminous.wait async
    // CHECK: {{.*}} = luminous.wait async [{{.*}}, {{.*}}]
    %t2 = luminous.wait async [%t0, %t1]
    // CHECK: luminous.wait [{{.*}}]
    luminous.wait [%t2]
    return
  }

  func @async_token(%arg0 : !luminous.async_token) -> !luminous.async_token {
    // CHECK-LABEL: func @async_token({{.*}}: !luminous.async_token)
    // CHECK: return {{.*}} : !luminous.async_token
    return %arg0 : !luminous.async_token
  }
}

# RUN: %PYTHON %s 2>&1

import ctypes
import gc, sys
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *

elm_wise_template = """
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {{
  func @function(%arg0: memref<{0}>, %arg1: memref<{1}>) -> memref<{2}> {{
    %0 = memref.alloc() : memref<{0}>
    linalg.generic {{indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%arg0, %arg1 : memref<{0}>, memref<{1}>) outs(%0 : memref<{2}>) {{
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %1 = addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }}
    return %0 : memref<{2}>
  }}
}}
"""

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

def run(f):
  log("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0

def transform(module, memref_size):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  pm = PassManager.parse('builtin.func(linalg-tile)')
  pm.run(module)
  return module

def testSameSizeElementWise():
  with Context():
    args = ["64x128x1024xf32"] * 3
    module = Module.parse(elm_wise_template.format(*args))
    module = transform(module, 100000)
    log(module)

run(testSameSizeElementWise)
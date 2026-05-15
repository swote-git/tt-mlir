// RUN: ttmlir-opt --convert-linalg-to-d2m -o %t %s
// RUN: FileCheck %s --input-file=%t

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func.func @generic_add
  func.func @generic_add(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %init = tensor.empty() : tensor<64x64xf32>
    // CHECK: d2m.to_layout
    // CHECK: d2m.generic
    // CHECK-SAME: iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: d2m.tile_add
    // CHECK: d2m.remote_store
    // CHECK: d2m.to_layout
    %0 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%init : tensor<64x64xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
      } -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @generic_exp
  func.func @generic_exp(%input: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %init = tensor.empty() : tensor<64x64xf32>
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: d2m.tile_exp
    %0 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<64x64xf32>)
        outs(%init : tensor<64x64xf32>) {
      ^bb0(%a: f32, %out: f32):
        %exp = math.exp %a : f32
        linalg.yield %exp : f32
      } -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @generic_relu_shape
  func.func @generic_relu_shape(%input: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %init = tensor.empty() : tensor<64x64xf32>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_fill
    // CHECK: d2m.tile_maximum
    %0 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<64x64xf32>)
        outs(%init : tensor<64x64xf32>) {
      ^bb0(%a: f32, %out: f32):
        %zero = arith.constant 0.000000e+00 : f32
        %relu = arith.maximumf %a, %zero : f32
        linalg.yield %relu : f32
      } -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}

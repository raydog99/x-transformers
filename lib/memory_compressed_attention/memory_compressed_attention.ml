open Torch

let conv_compress dim ~ratio ~groups =
  let conv =
    nn_conv1d
      ~padding:[]
      ~stride:ratio
      ~dilation:1
      ~groups
      ~bias:false
      ~output_channels:dim
      ~kernel_size:ratio
      ()
  in
  fun mem ->
    let mem_transposed = Tensor.transpose mem ~dim1:1 ~dim2:2 in
    let compressed_mem = conv (mem_padding ~value:0. mem_transposed) in
    Tensor.transpose compressed_mem ~dim1:1 ~dim2:2
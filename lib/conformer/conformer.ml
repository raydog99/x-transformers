open Torch

let swish = fun x -> Tensor.mul x (Tensor.sigmoid x)

let glu dim =
  fun x ->
    let out, gate = Tensor.chunk x ~chunks:2 ~dim in
    Tensor.mul out (Tensor.sigmoid gate)

let depthwise_conv1d chan_in chan_out kernel_size padding =
  let conv =
    nn_conv1d
      ~padding:(List.map (fun p -> (p, p)) padding)
      ~stride:1
      ~dilation:1
      ~groups:chan_in
      ~bias:false
      ~output_channels:chan_out
      ~kernel_size
      ()
  in
  fun x -> conv (nn_padding ~value:0. x)

let scale scale fn =
  fun x -> Tensor.mul (fn x) (Tensor.of_float scale)

let prenorm dim fn =
  let norm = nn_layer_norm ~eps:1e-5 dim in
  fun x ->
    let normalized = norm x in
    fn normalized
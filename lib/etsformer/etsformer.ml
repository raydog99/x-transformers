open Torch

let exists = function
  | Some _ | None -> true
  | _ -> false

let fourier_extrapolate signal start_ end_ =
  let device = T.device signal in
  let fhat = T.fft signal in
  let fhat_len = Array.length (T.shape fhat) in
  let time = T.linspace ~start:start_ ~end_:(end_ -. 1.) ~steps:(int_of_float (end_ -. start_)) ~device ~dtype:`ComplexFloat in
  let freqs = T.linspace ~start:0. ~end_:(float_of_int (fhat_len - 1)) ~steps:fhat_len ~device ~dtype:`ComplexFloat in
  let res = T.(fhat * (complex_float 0. 1. * (2. *. Float.pi * freqs *^ time) / float_of_int fhat_len |> exp)) / float_of_int fhat_len in
  T.sum res ~dim:[|-1|] |> T.real

let input_embedding time_features model_dim ~kernel_size ~dropout =
  nn_sequential [
    Rearrange ["b"; "n"; "d" --> "b"; "d"; "n"];
    nn_conv1d time_features model_dim ~kernel_size ~padding:(kernel_size / 2);
    nn_dropout dropout;
    Rearrange ["b"; "d"; "n" --> "b"; "n"; "d"]
  ]

let feed_forward dim ~mult ~dropout =
  nn_sequential [
    nn_linear dim (dim * mult);
    nn_sigmoid ();
    nn_dropout dropout;
    nn_linear (dim * mult) dim;
    nn_dropout dropout
  ]

let feed_forward_block ~dim ~mult ~dropout =
  let norm = LayerNorm.create dim in
  let ff = feed_forward dim ~mult ~dropout in
  let post_norm = LayerNorm.create dim in
  fun x ->
    let normalized = LayerNorm.forward norm x in
    LayerNorm.forward post_norm (T.(normalized + ff normalized))
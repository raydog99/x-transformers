open Base
open Torch

type residual = { fn : Tensor.t -> Tensor.t -> Tensor.t }

let residual fn x =
  let fn_out = fn x in
  Tensor.add fn_out x

type pre_norm = { norm : layer_norm; fn : Tensor.t -> Tensor.t }

let pre_norm dim fn x =
  let norm_x = layer_norm_forward x norm in
  fn norm_x

let geglU x =
  let x_chunks = Tensor.chunk x ~chunks:2 ~dim:(-1) in
  let x = List.nth_exn x_chunks 0 in
  let gates = List.nth_exn x_chunks 1 in
  Tensor.mul x (Tensor.gelu gates)

type feed_forward = { net : Tensor.t -> Tensor.t }

let feed_forward dim mult dropout =
  let inner_dim = dim * mult * 2 in
  let net = Sequential.of_list [
    Linear.create ~input_dim:dim ~output_dim:inner_dim;
    GEGLU.create ();
    Dropout.create ~p:dropout;
    Linear.create ~input_dim:inner_dim ~output_dim:dim;
  ] in
  { net }

let feed_forward_forward { net } x = Sequential.forward net x
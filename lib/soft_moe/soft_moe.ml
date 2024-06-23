open Torch
open Torch_nn
open Torch_functional

module LayerNorm = struct
  type t = { gamma : Tensor.t }

  let create dim =
    let gamma = Var.create ~shape:[|dim|] ~value:1. ~requires_grad:true in
    { gamma }

  let forward t x =
    let beta = Tensor.zeros_like t.gamma in
    F.layer_norm x ~normalized_shape:[|dim|] ~weight:t.gamma ~bias:beta
end

module RMSNorm = struct
  type t = { scale : float; gamma : Tensor.t }

  let create dim =
    let gamma = Var.create ~shape:[|dim|] ~value:1. ~requires_grad:true in
    { scale = sqrt (float_of_int dim); gamma }

  let forward t x =
    l2norm x * t.scale * t.gamma
end

let feed_forward dim mult dropout =
  let dim_hidden = int_of_float (float_of_int dim *. mult) in
  Sequential [
    Linear.create ~input_dim:dim ~output_dim:dim_hidden ();
    GELU.create ();
    Dropout.create ~p:dropout ();
    Linear.create ~input_dim:dim_hidden ~output_dim:dim ()
  ]

module GEGLU = struct
  let forward x =
    let x, gate = Tensor.chunk x ~chunks:2 ~dim:(-1) in
    x * F.gelu gate
end
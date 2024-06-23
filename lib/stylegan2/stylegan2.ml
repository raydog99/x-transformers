open Torch
open Torch_nn
open Torch_functional

module EMA = struct
  type t = { beta : float }

  let create beta = { beta }

  let update_average t old new =
    match old with
    | None -> new
    | Some old -> old * t.beta + (1. -. t.beta) * new
end

module Flatten = struct
  let forward x = Tensor.reshape x [| Tensor.size x.(0); -1 |]
end

module RandomApply = struct
  type t = { prob : float; fn : Tensor.t -> Tensor.t; fn_else : Tensor.t -> Tensor.t }

  let create prob fn fn_else = { prob; fn; fn_else }

  let forward t x =
    let fn = if Torch.rand () <= t.prob then t.fn else t.fn_else in
    fn x
end

module Residual = struct
  type t = { fn : Tensor.t -> Tensor.t }

  let create fn = { fn }

  let forward t x = t.fn x + x
end
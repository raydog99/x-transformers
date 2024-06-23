open Torch
open Tensor
open Layer
open Reduction

module Residual = struct
  type 'a t = {
    fn : 'a -> 'a;
  }

  let create fn = { fn }

  let forward t x =
    let fx = t.fn x in
    Tensor.(fx + x)
end

module RMSNorm = struct
  type t = {
    scale : float;
    gamma : Tensor.t;
  }

  let create dim =
    let scale = sqrt (float_of_int dim) in
    let gamma = Tensor.ones [dim; 1; 1; 1] in
    { scale; gamma }

  let forward t x =
    let normalized = Tensor.(normalize x ~dim:1 ~p:2.) in
    let scaled = Tensor.mul_scalar normalized t.scale in
    Tensor.(scaled * t.gamma)
end

module LayerNorm = struct
  type t = {
    gamma : Tensor.t;
  }

  let create dim =
    let gamma = Tensor.ones [1; dim; 1; 1; 1] in
    { gamma }

  let forward t x =
    let eps = if Tensor.dtype x = Torch.float32 then 1e-5 else 1e-3 in
    let var = Tensor.var x ~dim:1 ~unbiased:false ~keepdim:true in
    let mean = Tensor.mean x ~dim:1 ~keepdim:true in
    let normalized = Tensor.((x - mean) / ((var + eps) ** 0.5)) in
    Tensor.(normalized * t.gamma)
end

module WeightStandardizedConv3d = struct
  include Conv3d

  let forward t x =
    let eps = if Tensor.dtype x = Torch.float32 then 1e-5 else 1e-3 in

    let weight = t#weight in

    let mean = Tensor.(reduce weight ~dims:[|0|] Mean) in
    let var = Tensor.(reduce weight ~dims:[|0|] (fun w -> var w ~unbiased:false)) in
    let weight = Tensor.((weight - mean) * ((var + eps) ** (-0.5))) in

    Conv3d.forward_ t x
      ~weight
end
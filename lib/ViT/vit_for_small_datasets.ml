open Base
open Torch

module LSA = struct
  type t = {
    heads : int;
    temperature : Tensor.t;
    norm : nn.LayerNorm.t;
    attend : nn.Softmax.t;
    dropout : nn.Dropout.t;
    to_qkv : nn.Linear.t;
    to_out : nn.Sequential.t;
  }

  let create dim ~heads ~dim_head ~dropout =
    let inner_dim = dim_head * heads in
    let temperature = Tensor.log (Tensor.pow (Scalar.f 2.) (Scalar.f (-0.5))) in
    let norm = nn.LayerNorm dim in
    { heads; temperature; norm }
end
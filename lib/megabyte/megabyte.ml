open Base
open Torch

let token_shift t =
  let t, t_shift = Tensor.chunk t ~chunks:2 ~dim:(-1) in
  let t_shift = F.pad t_shift ~padding:(0, 0, 1, -1) ~value:0. in
  Tensor.cat ~dim:(-1) [t; t_shift]


module RotaryEmbedding = struct
  type t = {
    inv_freq : Tensor.t;
  }

  let create dim theta =
    let inv_freq = Tensor.arange ~end_:(F (float_of_int dim)) ~options:(Torch.DType Float) |> fun t ->
      Tensor.pow theta (Tensor.arange ~start:(F 0.) ~end_:(F (float_of_int dim)) ~options:(Torch.DType Float) / F (float_of_int dim)) |> Tensor.inv in
    { inv_freq }

  let device t = Tensor.device (Tensor.to_device t.inv_freq)

  let forward t seq_len =
    let t = Tensor.arange ~end_:(F (float_of_int seq_len)) ~options:(Torch.DType (Tensor.device t)) |> fun t ->
      Tensor.einsum "i,j->ij" [t; t.inv_freq] |> Tensor.cat ~dim:(-1) in
    Tensor.cat ~dim:(-1) [t; t]
end


let rotate_half x =
  let x1, x2 = Tensor.chunk x ~chunks:2 ~dim:(-1) in
  Tensor.cat ~dim:(-1) [Tensor.neg x2; x1]


let apply_rotary_pos_emb pos t =
  let sin_pos = Tensor.sin pos in
  let cos_pos = Tensor.cos pos in
  Tensor.(t * cos_pos + rotate_half t * sin_pos)
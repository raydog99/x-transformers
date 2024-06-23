open Torch

let min_expert_capacity = 4

let default val default_val =
  if isfunction default_val then default_val () else default_val

let cast_tuple el =
  if isinstance el tuple then el else (el,)

let top1 t =
  let values, index = T.topk t 1 ~dim:(-1) in
  let values = T.squeeze values ~dim:(-1) in
  let index = T.squeeze index ~dim:(-1) in
  values, index

let cumsum_exclusive t ?(dim=-1) =
  let num_dims = List.length (T.shape t) in
  let num_pad_dims = -dim - 1 in
  let pre_padding = List.init (2 * num_pad_dims) ~f:(fun _ -> 0) in
  let padded_t = T.pad t ~padding:(List.append pre_padding [1; 0]) |> T.cumsum ~dim in
  T.(padded_t.%[ []; slice None (-1) :: pre_padding ])

let safe_one_hot indexes max_length =
  let max_index = T.max indexes |> T.(+) (T.scalar 1) in
  let max_len = max max_index max_length in
  F.one_hot indexes max_len ~on_value:1 ~off_value:0

let init_ t =
  let dim = T.shape t |> List.last_exn in
  let std = 1. /. Float.sqrt (float dim) in
  T.uniform_ t ~a:(-.std) ~b:std

module GELU_ = struct
  let gelu_impl x =
    let sqrt_2_over_pi = Float.sqrt (2. /. Float.pi) in
    let c = 0.044715 in
    let pow3 = T.pow x (T.scalar 3.) in
    let tanh_part = T.tanh (sqrt_2_over_pi * (x + c * pow3)) in
    0.5 * x * (T.scalar 1. + tanh_part)
end

let gelu = if Torch.nn_hasattr "GELU" then Torch.nn.GELU else (module GELU_: Torch.nn.Module)
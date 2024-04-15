open Torch

(* Attention with Linear Bias *)
let alibi q k num_heads =
  let head_dim = Tensor.size2 q 2 in
  let slopes = Array.init num_heads (fun i -> 1. /. (2. ** float_of_int i)) in
  let bias = Array.init head_dim (fun i -> float_of_int (-i)) in
  let biases = Array.map (fun m -> Tensor.scalar_tensor (m *. float_of_int head_dim) * Tensor.of_array bias []) slopes in
  let biased_scores = Array.map2 (fun q_head k_head bias ->
    Tensor.add (Tensor.mm q_head (Tensor.transpose k_head 1 2)) bias)
    (Tensor.split_with_sizes q ~size:head_dim ~dim:2)
    (Tensor.split_with_sizes k ~size:head_dim ~dim:3)
    biases
  in
  Array.map Tensor.softmax biased_scores
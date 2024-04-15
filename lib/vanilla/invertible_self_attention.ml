open Torch

let normalize_cnn weights c =
  let weights_norm = Tensor.norm weights in
  if c < weights_norm then
    Tensor.mul_ weights (c /. weights_norm)
  else
    weights

let contractive_l2_mha q k v c n =
  let phi x = x *. exp(x +. 1.0) in
  let phi_inv x = Lambert.W0 (x /. exp 1.0) in
  let d_q, h = Array.dim1 q, Array.length q in
  (* The Lipschitz Constant of Self-Attention, Kim et. al*)
  let lipschitz_bound = sqrt (float_of_int n) *. sqrt (float_of_int d_q /. float_of_int h)
                        *. (phi_inv (float_of_int (n - 1) +. 1.0)) in
  let w_q, w_k, w_v, w_o = normalize_cnn q_weights c,
                          normalize_cnn k_weights c,
                          normalize_cnn v_weights c,
                          normalize_cnn o_weights (c /. lipschitz_bound) in
  let q_proj, k_proj, v_proj = Tensor.mm q w_q, Tensor.mm k w_k, Tensor.mm v w_v in
  let scores = Tensor.bmm q_proj k_proj.transpose(1, 2) in
  let attn = Tensor.softmax scores dim=-1 in
  let attn_vals = Tensor.bmm attn v_proj in
  Tensor.mm attn_vals w_o

let invertible_self_attn x =
  let f = contractive_l2_mha q k v c n in
  let y = Tensor.add x (Tensor.mul f 1.0) in
  let rec fixed_point y =
    let x_next = Tensor.sub y (Tensor.mul f x_next) in
    if Tensor.allclose x_next y then y else fixed_point x_next
  in
  fixed_point y
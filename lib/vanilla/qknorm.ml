open Torch

let qknorm q k g =
  let q_norm = Tensor.l2_normalize q ~dim:2 ~eps:1e-12 in
  let k_norm = Tensor.l2_normalize k ~dim:3 ~eps:1e-12 in
  Tensor.softmax (g * Tensor.bmm q_norm (Tensor.transpose k_norm 1 2))
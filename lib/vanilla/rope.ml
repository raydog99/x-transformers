open Torch

(* Rotary Position Embedding *)
let rope d =
  let d_2 = d / 2 in
  let thetas = Array.init d_2 (fun i -> 10000. ** (float_of_int (2 * (i + 1)) /. float_of_int d)) in
  let rows = Array.init d (fun i ->
    let row_idx = i / 2 in
    let col_idx = i mod 2 in
    let theta = thetas.(row_idx) in
    if col_idx = 0 then Tensor.cos theta
    else Tensor.sin theta)
  in
  Tensor.stack rows
(* On The Computational Complexity of Self-Attention, Keles et. al *)

open Torch

let compute_sv q k v p =
  let n, d_q, d_v = Array.dim q, Array.dim1 k, Array.dim2 v in
  let s = Array.make_matrix n n 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let s_ij = ref 0.0 in
      for z = 0 to p do
        s_ij := !s_ij +. C *. (Tensor.dot q.(i) k.(j)) ** float_of_int z
      done;
      s.(i).(j) <- !s_ij
    done
  done;
  let sv = Tensor.mm s v in
  sv

let compute_denominators q k p =
  let n, d_q = Array.dim q, Array.dim1 k in
  let denominators = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let sum = ref 0.0 in
    for j = 0 to n - 1 do
      for z = 0 to p do
        sum := !sum +. C *. (Tensor.dot q.(i) k.(j)) ** float_of_int z
      done
    done;
    denominators.(i) <- !sum
  done;
  denominators

(* Linear-time approximation of self-attention with Taylor series *)
let linear_self_attention q k v p =
  let sv = compute_sv q k v p in
  let denominators = compute_denominators q k p in
  let n, d_v = Array.dim sv, Array.dim2 sv in
  let result = Array.make_matrix n d_v 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to d_v - 1 do
      result.(i).(j) <- sv.(i).(j) /. denominators.(i)
    done
  done;
  result
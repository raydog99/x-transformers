open Base
open Torch

let conv1d_fft x weights ~dim ~weight_dim =
  let assert_condition cond =
    if not cond then
      failwith "Requires weight_dim >= dim"
  in
  assert_condition (weight_dim >= dim);

  let n = T.shape x.(dim) in
  let m = T.shape weights.(weight_dim) in

  let fast_len = Torch_fft.next_fast_len (n + m - 1) in

  let f_x = Torch_fft.rfft ~n:fast_len x ~dim in
  let f_weight = Torch_fft.rfft ~n:fast_len weights ~dim:weight_dim in

  let f_v_weight = T.(f_x * append_dims (T.conj f_weight) (weight_dim - dim)) in
  let out = Torch_fft.irfft f_v_weight ~n:fast_len ~dim in
  let out = T.roll out ~shifts:(-1) ~dims:[|dim|] in

  let indices = T.arange ~start:(fast_len - n) ~end_:fast_len ~dtype:`Long ~device:(T.device x) in
  T.index_select out dim indices

let t5_relative_position_bias scale ~causal ~num_buckets ~max_distance =
  let relative_attention_bias = nn_embedding num_buckets 1 in
  let relative_position_bucket relative_position ~causal ~num_buckets ~max_distance =
    let n = T.neg relative_position in
    let num_buckets = if causal then num_buckets else num_buckets / 2 in
    let ret = T.(zero_like n * (n < zero_like n) * num_buckets) in
    let n = T.abs n in
    let max_exact = num_buckets / 2 in
    let is_small = T.(n < max_exact) in
    let val_if_large =
      let log_val = T.log (n / max_exact) / T.log (max_distance / max_exact) in
      T.min (max_exact + (log_val * (num_buckets - max_exact) |> T.long)) (num_buckets - 1)
    in
    let val_if_large = T.where is_small n val_if_large in
    T.(ret + val_if_large)
  in
  fun x ->
    let i, j, device = T.shape x.(-2), T.shape x.(-1), T.device x in
    let q_pos = T.arange ~start:0 ~end_:i ~dtype:`Long ~device in
    let k_pos = T.arange ~start:0 ~end_:j ~dtype:`Long ~device in
    let rel_pos = T.(rearrange k_pos "j -> 1 j" - rearrange q_pos "i -> i 1") in
    let rp_bucket = relative_position_bucket rel_pos ~causal ~num_buckets ~max_distance in
    let values = relative_attention_bias rp_bucket in
    let bias = T.rearrange values "i j 1 -> i j" in
    T.(bias * scale)
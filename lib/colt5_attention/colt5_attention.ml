open Core

let cumsum_exclusive t ~dim =
  assert (dim < 0);
  let num_pad_dims = (-dim) - 1 in
  let pre_padding = Array.init (2 * num_pad_dims) (fun _ -> 0) in
  let pad_spec = Array.append pre_padding [|1; -1|] in
  let padded_t = Array.append t pad_spec in
  padded_t

let log t ~eps =
  let clipped_t = if t < eps then eps else t in
  Float.log clipped_t

let gumbel_noise t =
  let noise = Array.init (Array.length t) (fun _ -> Random.float 1.0) in
  Array.map (fun x -> -.(log (-.(log x)))) noise

let safe_one_hot indexes max_length =
  let max_index = Array.fold_left max 0 indexes + 1 in
  let one_hot_classes = max (max_index + 1) max_length in
  let encoded = Array.make_matrix (Array.length indexes) one_hot_classes 0 in
  encoded
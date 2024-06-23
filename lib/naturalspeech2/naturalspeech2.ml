open Torch
open Base

let pad_or_curtail_to_length t length =
  if Tensor.shape t |> Array.last_exn = length then t
  else if Tensor.shape t |> Array.last_exn > length then Tensor.narrow t ~dim:(-1) ~start:0 ~length
  else F.pad t ~padding:(Torch.PaddingSpec.create ~pad_l:0 ~pad_r:(length - Tensor.shape t |> Array.last_exn) ())

let prob_mask_like shape prob device =
  if Float.(prob = 1.) then Torch.ones ~device ~dtype:Bool shape
  else if Float.(prob = 0.) then Torch.zeros ~device ~dtype:Bool shape
  else Torch.rand shape ~device ~dtype:Float32 |> fun rand -> Tensor.(rand < prob)

let generate_mask_from_repeats repeats =
  let repeats = Tensor.to_int repeats in
  let device = Tensor.device repeats in
  let lengths = Tensor.sum repeats ~dim:(-1) in
  let max_length = Tensor.amax lengths |> Tensor.item in
  let cumsum = Tensor.cumsum repeats ~dim:(-1) in
  let cumsum_exclusive = F.pad cumsum ~padding:(Torch.PaddingSpec.create ~pad_l:1 ~pad_r:(-1) ~value:0.) in
  let seq = Tensor.arange ~device max_length in
  let seq = Tensor.repeat_ seq ~repeats:(List.append (List.init (Tensor.shape repeats).(0) ~f:(fun _ -> 1)) [1]) in
  let cumsum = Tensor.unsqueeze cumsum ~dim:(-1) in
  let cumsum_exclusive = Tensor.unsqueeze cumsum_exclusive ~dim:(-1) in
  let lengths = Tensor.unsqueeze lengths ~dim:1 in
  let mask = Tensor.(seq < cumsum && seq >= cumsum_exclusive && seq < lengths) in
  mask

let learned_sinusoidal_pos_emb dim =
  assert (dim mod 2 = 0);
  let half_dim = dim / 2 in
  let weights = nn_parameter (Tensor.randn [half_dim]) in
  fun x ->
    let x = Tensor.unsqueeze x ~dim:1 in
    let freqs = Tensor.(x * Tensor.unsqueeze weights ~dim:0 * 2. * Float.pi) in
    let fouriered = Tensor.cat [Tensor.sin freqs; Tensor.cos freqs] ~dim:(-1) in
    Tensor.cat [x; fouriered] ~dim:(-1)

let compute_pitch_pytorch wav sample_rate =
  let pitch_feature = Torchaudio.Functional.compute_kaldi_pitch wav sample_rate in
  let pitch, _ = Tensor.unbind pitch_feature ~dim:(-1) in
  pitch
open Torch

type pre_norm = {
  fn : Tensor.t -> kwargs:'a -> Tensor.t;
  norm : nn.LayerNorm.t;
}

let create_pre_norm (dim : int) (fn : Tensor.t -> kwargs:'a -> Tensor.t) : pre_norm =
  {
    fn;
    norm = nn.LayerNorm.create dim;
  }

let forward_pre_norm (x : Tensor.t) (kwargs : 'a) (pn : pre_norm) : Tensor.t =
  let x_norm = pn.norm |> forward_layer_norm x in
  pn.fn x_norm ~kwargs

type feed_forward = {
  net : nn.Sequential.t;
}

let create_feed_forward (dim : int) ~mult : feed_forward =
  let dim_mult = dim * mult in
  let net =
    nn.Sequential.of_list
      [
        nn.Linear.create dim dim_mult;
        nn.GELU.create ();
        nn.Linear.create dim_mult dim;
      ]
  in
  { net }

let forward_feed_forward (x : Tensor.t) (ff : feed_forward) : Tensor.t =
  nn.Sequential.forward x ff.net

type pre_shift_tokens = {
  fn : Tensor.t -> kwargs:'a -> Tensor.t;
  shifts : int list;
}

let create_pre_shift_tokens (shifts : int list) (fn : Tensor.t -> kwargs:'a -> Tensor.t) : pre_shift_tokens =
  { fn; shifts = List.to_seq shifts |> Tuple.of_seq }

let forward_pre_shift_tokens (x : Tensor.t) (kwargs : 'a) (pst : pre_shift_tokens) : Tensor.t =
  let mask = kwargs |> List.assoc_opt "mask" in
  let segments = List.length pst.shifts in
  let feats_per_shift = Tensor.shape x |> List.last_exn // segments in
  let splitted = Tensor.split x ~split_size:[feats_per_shift] ~dim:(-1) in
  let segments_to_shift =
    List.map
      (fun (seg, shift) -> shift_token seg shift ~mask)
      (List.zip_exn (List.take splitted segments) pst.shifts)
  in
  Tensor.cat (Array.of_list (segments_to_shift @ List.drop splitted segments)) ~dim:(-1)
  |> pst.fn ~kwargs
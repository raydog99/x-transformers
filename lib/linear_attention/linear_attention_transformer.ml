open Torch
open Tensor
open Layer

let shift t amount mask =
  if amount = 0 then t
  else
    let t = match mask with
      | Some m -> Tensor.masked_fill t ~mask:(Tensor.logical_not m) ~value:0.0
      | None -> t
    in
    Tensor.pad t ~pad:[| 0; 0; amount; -amount |] ~value:0.0

module PreShiftTokens = struct
  type t = {
    fn: Tensor.t -> Tensor.t;
    shifts: int array;
  }

  let create ~shifts ~fn = {
    fn;
    shifts = Array.of_list shifts;
  }

  let forward t x ?mask () =
    let segments = Array.length t.shifts in
    let feats_per_shift = Tensor.shape x |> List.hd |> ( / ) segments in
    let splitted = Tensor.split x ~split_size:feats_per_shift ~dim:(-1) in
    let segments_to_shift = Array.sub splitted 0 segments |> Array.to_list in
    let rest = Array.sub splitted segments (Array.length splitted - segments) |> Array.to_list in
    let segments_to_shift = List.map2 (fun s shift -> shift s shift mask) segments_to_shift t.shifts in
    Tensor.cat (List.append segments_to_shift rest) ~dim:(-1) |> t.fn
end

module AbsolutePositionalEmbedding = struct
  type t = {
    emb: Layer.t;
  }

  let create ~dim ~max_seq_len =
    { emb = Layer.embedding ~num_embeddings:max_seq_len ~embedding_dim:dim }

  let forward t x =
    let t = Tensor.arange1 ~end_:Tensor.(float (List.nth (Tensor.shape x) 1)) ~dtype:Tensor.int64 ~device:(Tensor.device x) in
    Layer.forward t.emb t |> Tensor.unsqueeze ~dim:0
end

module FixedPositionalEmbedding = struct
  type t = {
    emb: Tensor.t;
  }

  let create ~dim ~max_seq_len =
    let inv_freq = Tensor.(div_scalar1 (float 1.) (pow_scalar (float 10000.) (Tensor.div_scalar (arange0 (float 0.) (float dim) 2.) (float dim)))) in
    let position = Tensor.arange0 ~end_:Tensor.(float max_seq_len) ~dtype:Tensor.float in
    let sinusoid_inp = Tensor.einsum2 "i,j->ij" position inv_freq in
    let emb = Tensor.cat2 [|Tensor.sin sinusoid_inp; Tensor.cos sinusoid_inp|] ~dim:(-1) in
    { emb }

  let forward t x =
    Tensor.narrow t.emb ~dim:0 ~start:0 ~length:(List.nth (Tensor.shape x) 1)
end

let rotate_every_two x =
  let x = Tensor.view x ~size:[-1; 2] in
  let x1, x2 = Tensor.unbind2 x ~dim:(-1) in
  let x = Tensor.stack2 [|Tensor.neg x2; x1|] ~dim:(-1) in
  Tensor.view x ~size:(Tensor.shape x |> List.hd |> fun s -> [s * 2])

let apply_rotory_pos_emb q k sinu_pos =
  let sinu_pos = Tensor.view sinu_pos ~size:[List.hd (Tensor.shape sinu_pos); 2; -1] in
  let sin, cos = Tensor.unbind2 sinu_pos ~dim:(-2) in
  let sin, cos = Tensor.(repeat sin ~dim:[0; -1; 2]), Tensor.(repeat cos ~dim:[0; -1; 2]) in
  let q = Tensor.add (Tensor.mul q cos) (Tensor.mul (rotate_every_two q) sin) in
  let k = Tensor.add (Tensor.mul k cos) (Tensor.mul (rotate_every_two k) sin) in
  q, k

module GELU_ = struct
  let forward x =
    Tensor.(mul_scalar1 (mul_scalar (add_scalar (pow x (float 3.)) (float 0.044715)) (sqrt (float (2. /. Float.pi)))) 0.5)
end

let gelu x = if Layer.has_gelu then Tensor.gelu x else GELU_.forward x
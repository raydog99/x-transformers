open Torch
open Tensor
open Layer

module SinusoidalPosEmb = struct
  type t = {
    dim : int;
    theta : float;
  }

  let create ~dim ?(theta=10000.) () =
    { dim; theta }

  let forward t x =
    let dtype, device = Tensor.dtype x, Tensor.device x in
    assert (dtype = Torch.float32) "input to sinusoidal pos emb must be a float type";

    let half_dim = t.dim / 2 in
    let emb = (log t.theta) /. float_of_int (half_dim - 1) in
    let emb = Tensor.exp (Tensor.arange0 ~end_:(float_of_int half_dim) ~device ~dtype:Float) * (-. emb) in
    let emb = Tensor.unsqueeze (Tensor.rearrange x ~dims:"i" ~axis:[||]) (-1) * Tensor.unsqueeze emb 0 in
    let emb = Tensor.cat [Tensor.sin emb; Tensor.cos emb] ~dim:(-1) in
    Tensor.to_type emb dtype
end

module RMSNorm = struct
  type t = {
    chan : int;
    dim : int;
    gamma : Tensor.t;
  }

  let create ~chan ?(dim=1) () =
    let gamma = Tensor.ones [chan] in
    { chan; dim; gamma }

  let forward t x =
    let dim = t.dim in
    let right_ones = if dim < 0 then (dim + 1) else (Tensor.ndim x - 1 - dim) in
    let gamma = Tensor.reshape t.gamma (List.append [-1] (List.init right_ones (fun _ -> 1))) in
    let scale = sqrt (float_of_int (Tensor.shape x).(dim)) in
    Tensor.(normalize x ~dim ~p:2.) * scale * gamma
end

let shift_token t =
  let t, t_shift = Tensor.chunk t ~chunks:2 ~dim:1 in
  let t_shift = Tensor.pad t_shift ~padding:[0; 0; 1; -1] ~value:0. in
  Tensor.cat [t; t_shift] ~dim:1

module GEGLU = struct
  type t = unit

  let create () = ()

  let forward () x =
    let x, gate = Tensor.chunk x ~chunks:2 ~dim:1 in
    x * Tensor.gelu gate
end
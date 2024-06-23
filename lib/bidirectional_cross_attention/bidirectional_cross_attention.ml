open Torch
open Tensor
open Layer

let default opt default_val =
  match opt with
  | Some v -> v
  | None -> default_val

let exists opt =
  match opt with
  | Some _ -> true
  | None -> false

module BidirectionalCrossAttention = struct
  type t = {
    norm: Layer.t;
    context_norm: Layer.t;
    heads: int;
    scale: float;
    dropout: Layer.t;
    context_dropout: Layer.t;
    to_qk: Layer.t;
    context_to_qk: Layer.t;
    to_v: Layer.t;
    context_to_v: Layer.t;
    to_out: Layer.t;
    context_to_out: Layer.t;
    talking_heads: Layer.t;
    context_talking_heads: Layer.t;
  }

  let create ~dim ?(heads=8) ?(dim_head=64) ?context_dim ?(dropout=0.) ?(talking_heads=false) ?(prenorm=false) () =
    let context_dim = default context_dim dim in
    let norm = if prenorm then Layer.layer_norm dim else Layer.id in
    let context_norm = if prenorm then Layer.layer_norm context_dim else Layer.id in
    let scale = 1. /. (sqrt (float dim_head)) in
    let inner_dim = dim_head * heads in
    let dropout = Layer.dropout ~p:dropout in
    let context_dropout = Layer.dropout ~p:dropout in
    let to_qk = Layer.linear ~input_dim:dim inner_dim in
    let context_to_qk = Layer.linear ~input_dim:context_dim inner_dim in
    let to_v = Layer.linear ~input_dim:dim inner_dim in
    let context_to_v = Layer.linear ~input_dim:context_dim inner_dim in
    let to_out = Layer.linear ~input_dim:inner_dim dim in
    let context_to_out = Layer.linear ~input_dim:inner_dim context_dim in
    let talking_heads = if talking_heads then Layer.conv2d ~ksize:1 ~input_dim:heads heads else Layer.id in
    let context_talking_heads = if talking_heads then Layer.conv2d ~ksize:1 ~input_dim:heads heads else Layer.id in
    { norm; context_norm; heads; scale; dropout; context_dropout; to_qk; context_to_qk; to_v; context_to_v; to_out; context_to_out; talking_heads; context_talking_heads }

  let forward t x context ?mask ?context_mask ?(return_attn=false) ?rel_pos_bias () =
    let b = Tensor.shape x |> List.hd in
    let i = Tensor.shape x |> List.nth (-2) in
    let j = Tensor.shape context |> List.nth (-2) in
    let device = Tensor.device x in

    let x = Layer.forward t.norm x in
    let context = Layer.forward t.context_norm context in

    let qk = Layer.forward t.to_qk x in
    let v = Layer.forward t.to_v x in
    let context_qk = Layer.forward t.context_to_qk context in
    let context_v = Layer.forward t.context_to_v context in

    let qk, context_qk, v, context_v = 
      List.map (fun t -> Tensor.view t ~shape:[b; t.heads; -1; t.dim_head]) [qk; context_qk; v; context_v] 
    in

    let sim = Tensor.(einsum "b h i d, b h j d -> b h i j" [qk; context_qk]) * t.scale in

    let sim = 
      match rel_pos_bias with
      | Some bias -> Tensor.( + ) sim bias
      | None -> sim
    in

    let sim =
      match mask, context_mask with
      | Some m, Some cm -> 
          let attn_mask = Tensor.(reshape m [b; 1; i; 1]) * Tensor.(reshape cm [b; 1; 1; j]) in
          Tensor.masked_fill sim ~mask:(Tensor.logical_not attn_mask) ~value:Tensor.(finfo sim.float_t).max_neg
      | Some m, None ->
          let attn_mask = Tensor.(reshape m [b; 1; i; 1]) in
          Tensor.masked_fill sim ~mask:(Tensor.logical_not attn_mask) ~value:Tensor.(finfo sim.float_t).max_neg
      | None, Some cm ->
          let attn_mask = Tensor.(reshape cm [b; 1; 1; j]) in
          Tensor.masked_fill sim ~mask:(Tensor.logical_not attn_mask) ~value:Tensor.(finfo sim.float_t).max_neg
      | None, None -> sim
    in

    let attn = Tensor.softmax sim ~dim:(-1) in
    let context_attn = Tensor.softmax sim ~dim:(-2) in

    let attn = Layer.forward t.dropout attn in
    let context_attn = Layer.forward t.context_dropout context_attn in

    let attn = Layer.forward t.talking_heads attn in
    let context_attn = Layer.forward t.context_talking_heads context_attn in

    let out = Tensor.einsum "b h i j, b h j d -> b h i d" [attn; context_v] in
    let context_out = Tensor.einsum "b h j i, b h j d -> b h i d" [context_attn; v] in

    let out = Tensor.view out ~shape:[b; -1; t.heads * t.dim_head] in
    let context_out = Tensor.view context_out ~shape:[b; -1; t.heads * t.dim_head] in

    let out = Layer.forward t.to_out out in
    let context_out = Layer.forward t.context_to_out context_out in

    if return_attn then
      out, context_out, attn, context_attn
    else
      out, context_out
end
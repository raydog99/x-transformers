open Base

module RMSNorm : sig
  type t

  val create : int -> t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    scale : float;
    gamma : Tensor.t;
  }

  let create dim =
    let scale = sqrt (float dim) in
    let gamma = Tensor.ones [|dim|] in
    { scale; gamma }

  let forward self x =
    let normalized_x = Tensor.normalize x ~dim:(-1) in
    Tensor.mul normalized_x (Tensor.mul_scalar self.gamma self.scale)
end

module FeedForward : sig
  type t

  val create : int -> mult:int -> dropout:float -> t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    net : Module.t;
  }

  let create dim ~mult ~dropout =
    let dim_inner = dim * mult in
    let net =
      Sequential.create
        [
          RMSNorm.create dim;
          Linear.create dim dim_inner;
          GELU.create ();
          Dropout.create dropout;
          Linear.create dim_inner dim;
          Dropout.create dropout;
        ]
    in
    { net }

  let forward self x = Sequential.forward self.net x
end

module Attention : sig
  type t

  val create :
    int ->
    heads:int ->
    dim_head:int ->
    dropout:float ->
    rotary_embed:Module.t option ->
    flash:bool ->
    t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    heads : int;
    scale : float;
    attend : Module.t;
    norm : RMSNorm.t;
    to_qkv : Linear.t;
    to_gates : Linear.t;
    to_out : Sequential.t;
    rotary_embed : Module.t option;
  }

  let create dim ~heads ~dim_head ~dropout ~rotary_embed ~flash =
    let scale = 1. /. sqrt (float dim_head) in
    let dim_inner = heads * dim_head in
    let attend = Attend.create ~flash ~dropout in
    let norm = RMSNorm.create dim in
    let to_qkv = Linear.create dim (dim_inner * 3) ~bias:false in
    let to_gates = Linear.create dim heads in
    let to_out =
      Sequential.create
        [
          Linear.create dim_inner dim ~bias:false;
          Dropout.create dropout;
        ]
    in
    { heads; scale; attend; norm; to_qkv; to_gates; to_out; rotary_embed }

  let forward self x =
    let x = RMSNorm.forward self.norm x in
    let qkv = Linear.forward self.to_qkv x |> Tensor.rearrange ~pattern:"b n (qkv h d) -> qkv b h n d" ~qkv:3 ~h:self.heads in
    let q, k, v = Tensor.split qkv ~split_sizes:[|[|dim_inner; dim_inner; dim_inner|]|] ~dim:2 in
    let q, k =
      match self.rotary_embed with
      | Some rotary_embed ->
          let rotate = RotaryEmbed.rotate_queries_or_keys rotary_embed in
          rotate q, rotate k
      | None -> q, k
    in
    let out = Attend.forward self.attend q k v in
    let gates = Linear.forward self.to_gates x in
    let out = Tensor.mul out (Tensor.rearrange gates ~pattern:"b n h -> b h n 1" |> Tensor.sigmoid) in
    Sequential.forward self.to_out (Tensor.rearrange out ~pattern:"b h n d -> b n (h d)")
end
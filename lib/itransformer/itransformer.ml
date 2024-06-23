module Attention : sig
  type t

  val create :
    dim:int ->
    dim_head:int ->
    heads:int ->
    dropout:float ->
    flash:bool ->
    t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    scale : float;
    to_qkv : Module.t;
    to_v_gates : Module.t;
    attend : Attend.t;
    to_out : Module.t;
  }

  let create ~dim ~dim_head ~heads ~dropout ~flash =
    let scale = 2. ** (-0.5) *. float dim_head in
    let dim_inner = dim_head * heads in
    let to_qkv = Sequential.create [
      Linear.create dim (dim_inner * 3) ~bias:false;
      Rearrange.create "b n (qkv h d) -> qkv b h n d" ~qkv:3 ~h:heads;
    ] in
    let to_v_gates = Sequential.create [
      Linear.create dim heads ~bias:false;
      Sigmoid.create;
      Rearrange.create "b n h -> b h n 1" ~h:heads;
    ] in
    let attend = Attend.create ~flash ~dropout in
    let to_out = Sequential.create [
      Rearrange.create "b h n d -> b n (h d)";
      Linear.create dim_inner dim ~bias:false;
      Dropout.create dropout;
    ] in
    { scale; to_qkv; to_v_gates; attend; to_out }

  let forward self x =
    let qkv = Module.forward self.to_qkv x in
    let q, k, v = Rearrange.forward (Rearrange.create "b n (qkv h d) -> qkv b h n d" ~qkv:3 ~h:heads) qkv in
    let out = Attend.forward self.attend q k v in
    let gates = Module.forward self.to_v_gates x in
    let out = Tensor.mul out gates in
    Module.forward self.to_out out
end

let feed_forward dim mult dropout =
  let dim_inner = int_of_float (float dim *. mult *. 2. /. 3.) in
  Sequential.create [
    Linear.create dim (dim_inner * 2);
    GEGLU.create;
    Dropout.create dropout;
    Linear.create dim_inner dim;
  ]


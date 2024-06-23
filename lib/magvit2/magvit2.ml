module ToTimeSequence : sig
  type t

  val create : Module.t -> t

  val forward : t -> Tensor.t -> kwargs:(string * Tensor.t) list -> Tensor.t
end = struct
  type t = {
    fn : Module.t;
  }

  let create fn = { fn }

  let forward self x ~kwargs =
    let x = Rearrange.rearrange x "b c f ... -> b ... f c" in
    let x, ps = Pack.pack_one x "* n c" in
    let o = Module.forward self.fn x ~kwargs in
    let o = Pack.unpack_one o ps "* n c" in
    Rearrange.rearrange o "b ... f c -> b c f ..."
end

module SqueezeExcite : sig
  type t

  val create :
    int ->
    dim_out:int option ->
    dim_hidden_min:int ->
    init_bias:float ->
    t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    to_k : Module.t;
    net : Module.t;
  }

  let create dim ~dim_out ~dim_hidden_min ~init_bias =
    let dim_out = Option.value dim_out ~default:dim in
    let to_k = Conv2d.create dim 1 1 in
    let dim_hidden = max dim_hidden_min (dim_out / 2) in
    let net =
      Sequential.create
        [
          Conv2d.create dim dim_hidden 1;
          LeakyReLU.create 0.1;
          Conv2d.create dim_hidden dim_out 1;
          Sigmoid.create;
        ]
    in
    nn_init.zeros_ net.(1).weight;
    nn_init.constant_ net.(1).bias init_bias;
    { to_k; net }

  let forward self x =
    let orig_input = x in
    let batch = Tensor.shape x |> Array.get 0 in
    let is_video = Tensor.ndim x = 5 in
    let x =
      if is_video then Rearrange.rearrange x "b c f h w -> (b f) c h w" else x
    in
    let context = Module.forward self.to_k x in
    let context = Rearrange.rearrange context "b c h w -> b c (h w)" |> Tensor.softmax ~dim:(-1) in
    let spatial_flattened_input = Rearrange.rearrange x "b c h w -> b c (h w)" in
    let out = Tensor.einsum "b i n, b c n -> b c i" context spatial_flattened_input in
    let out = Rearrange.rearrange out "... -> ... 1" in
    let gates = Module.forward self.net out in
    let gates =
      if is_video then Rearrange.rearrange gates "(b f) c h w -> b c f h w" ~b:batch else gates
    in
    Tensor.mul gates orig_input
end

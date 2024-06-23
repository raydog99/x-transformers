open Base

module MultiHeadRMSNorm : sig
  type t

  val create : int -> heads:int -> t

  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    scale : float;
    gamma : Tensor.t;
  }

  let create dim ~heads =
    let scale = sqrt (float dim) in
    let gamma = nn_init.ones_ (Tensor.create [|heads; 1; dim|]) in
    { scale; gamma }

  let forward self x =
    let normalized_x = Tensor.normalize x ~dim:(-1) in
    Tensor.mul (Tensor.mul normalized_x self.gamma) self.scale
end

module JointAttention : sig
  type t

  val create :
    dim:int ->
    dim_inputs:int array ->
    dim_head:int ->
    heads:int ->
    qk_rmsnorm:bool ->
    flash:bool ->
    attend_kwargs:(string * Tensor.t) list ->
    t

  val forward : t -> Tensor.t array -> masks:Tensor.t array option -> Tensor.t array
end = struct
  type t = {
    num_inputs : int;
    to_qkv : Module.t array;
    split_heads : Rearrange.t;
    attend : Attend.t;
    merge_heads : Rearrange.t;
    to_out : Module.t array;
    q_rmsnorms : MultiHeadRMSNorm.t array;
    k_rmsnorms : MultiHeadRMSNorm.t array;
    qk_rmsnorm : bool;
    dummy : Tensor.t;
  }

  let create ~dim ~dim_inputs ~dim_head ~heads ~qk_rmsnorm ~flash ~attend_kwargs =
    let dim_inner = dim_head * heads in
    let num_inputs = Array.length dim_inputs in
    let to_qkv =
      Array.init num_inputs (fun i -> Linear.create dim (dim_inner * 3) ~bias:false)
    in
    let split_heads = Rearrange.create "b n (qkv h d) -> qkv b h n d" ~h:heads in
    let attend = Attend.create ~flash ~kwargs:attend_kwargs in
    let merge_heads = Rearrange.create "b h n d -> b n (h d)" in
    let to_out = Array.init num_inputs (fun i -> Linear.create dim_inner dim_inputs.(i) ~bias:false) in
    let q_rmsnorms =
      if qk_rmsnorm then Array.init num_inputs (fun i -> MultiHeadRMSNorm.create dim_head ~heads) else [||]
    in
    let k_rmsnorms =
      if qk_rmsnorm then Array.init num_inputs (fun i -> MultiHeadRMSNorm.create dim_head ~heads) else [||]
    in
    let dummy = Tensor.create_empty () in
    { num_inputs; to_qkv; split_heads; attend; merge_heads; to_out; q_rmsnorms; k_rmsnorms; qk_rmsnorm; dummy }

  let forward self inputs ~masks =
    let device = Tensor.device self.dummy in
    assert (Array.length inputs = self.num_inputs);
    let masks = Option.value masks ~default:(Array.make self.num_inputs (Tensor.create_empty ())) in
    let all_qkvs = Array.map2_exn inputs masks self.to_qkv self.q_rmsnorms self.k_rmsnorms ~f:(fun x mask to_qkv q_rmsnorm k_rmsnorm ->
      let qkv = Module.forward to_qkv x in
      let qkv = Rearrange.forward self.split_heads qkv in
      if self.qk_rmsnorm then (
        let q, k, v = qkv in
        let q = MultiHeadRMSNorm.forward q_rmsnorm q in
        let k = MultiHeadRMSNorm.forward k_rmsnorm k in
        [|q; k; v|]
      ) else [|qkv|]
    ) in
    let all_qkvs, packed_shape = Pack.pack all_qkvs "qkv b h * d" in
    let q, k, v = Array.to_list all_qkvs in
    let outs, _, _ = Attend.forward self.attend q k v ~mask:(Pack.pack masks "b *") in
    let outs = Rearrange.forward self.merge_heads outs in
    let outs = Pack.unpack outs packed_shape "b * d" in
    Array.map2_exn outs self.to_out ~f:(fun out to_out ->
      Module.forward to_out out
    )
end
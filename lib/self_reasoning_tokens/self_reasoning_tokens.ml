open Base

module CausalAttention : sig
  type t

  val create : int -> dim_head:int -> heads:int -> t

  val forward :
    t ->
    Tensor.t ->
    attn_mask:Tensor.t option ->
    stop_grad_attn_mask:(Tensor.t * Tensor.t * Tensor.t) option ->
    Tensor.t
end = struct
  type t = {
    scale : float;
    to_qkv : Module.t;
    to_out : Module.t;
    heads : int;
  }

  let create dim ~dim_head ~heads =
    let scale = 1. /. sqrt (float dim_head) in
    let dim_inner = dim_head * heads in
    let to_qkv =
      Sequential.create
        [
          RMSNorm.create dim;
          Linear.create dim (dim_inner * 3) ~bias:false;
          Rearrange.create "b n (qkv h d) -> qkv b h n d" ~qkv:3 ~h:heads;
        ]
    in
    let to_out =
      Sequential.create
        [
          Rearrange.create "b h n d -> b n (h d)";
          Linear.create dim_inner dim ~bias:false;
        ]
    in
    { scale; to_qkv; to_out; heads }

  let forward self x ~attn_mask ~stop_grad_attn_mask =
    let seq = Tensor.shape x |> Array.last in
    let q, k, v = Sequential.forward self.to_qkv x in
    let out =
      match stop_grad_attn_mask with
      | Some (q_stop_grad, k_stop_grad, v_stop_grad) ->
          stop_graddable_attn q k v ~attn_mask ~q_stop_grad_mask:q_stop_grad ~k_stop_grad_mask:k_stop_grad ~v_stop_grad_mask:v_stop_grad
      | None ->
          let q = Tensor.mul_scalar q self.scale in
          let sim = Tensor.einsum q k ~equation:"b h i d, b h j d -> b h i j" in
          let causal_mask = Tensor.ones [|seq; seq|] ~device:Device.cuda ~dtype:Dtype.bool_ |> Tensor.triu ~diagonal:1 in
          let mask_value = Tensor.finfo sim |> Tensor.max_value |> Tensor.neg in
          let sim = Tensor.masked_fill sim ~mask:causal_mask ~value:mask_value in
          let sim =
            match attn_mask with
            | Some mask -> Tensor.masked_fill sim ~mask:(Tensor.logical_not mask) ~value:mask_value
            | None -> sim
          in
          let attn = Tensor.softmax sim ~dim:(-1) in
          Tensor.einsum attn v ~equation:"b h i j, b h j d -> b h i d"
    in
    Sequential.forward self.to_out out
end
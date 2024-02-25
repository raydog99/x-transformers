open Torch

module MultiheadAttention : sig
  type t

  val attention :
    query:Tensor.t ->
    key:Tensor.t ->
    value:Tensor.t ->
    attention_Mask:Tensor.t option ->
    is_training:bool ->
    Tensor.t

  val split_heads :
    xs:Tensor.t ->
    k:bool ->
    Tensor.t

  val create :
    vs:Torch_nn.Parameter.t list ->
    config:Config.t ->
    scale:bool ->
    xs:Tensor.t ->
    layer_past:Tensor.t option ->
    attention_mask:Tensor.t option ->
    is_training:bool ->
    Tensor.t * Tensor.t
end
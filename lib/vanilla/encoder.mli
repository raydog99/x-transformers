open Torch

module Encoder : sig
  type t

  val layer :
    vs:Var_store.t ->
    config:Config.t ->
    scale:bool ->
    xs:Tensor.t ->
    layer_past:Tensor.t option ->
    attention_mask:Tensor.t option ->
    is_training:bool ->
    Tensor.t * Tensor.t

  val create :
    vs:Var_store.t ->
    config:Config.t ->
    t list
end
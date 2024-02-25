open Base
open Torch

module Transformer : sig
  type t

  val create :
    vs:Var_store.t ->
    config:Config.t ->
    t

  val input_ids :
    t ->
    layer_past:Tensor.t option ->
    attention_mask:Tensor.t option ->
    token_type_ids:Tensor.t option ->
    position_ids:Tensor.t option ->
    is_training:bool ->
    Tensor.t * [ `layer of Tensor.t array ]
end
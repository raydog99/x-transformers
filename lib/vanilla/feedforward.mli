open Torch

module Feedforward : sig
  val create :
    config:Config.t ->
    Tensor.t ->
    is_training:bool ->
    Tensor.t
end
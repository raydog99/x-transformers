open Base
open Torch

module NaViT : sig
  type t

  val create : vs:Var_store.t -> config:Config.t -> t

  val forward :
    t ->
    input_images:Tensor.t list ->
    group_images:bool ->
    group_max_seq_len:int ->
    Tensor.t
end

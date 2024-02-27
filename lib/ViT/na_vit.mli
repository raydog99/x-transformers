open Base
open Torch

module NaViT : sig
  type t

  val create : vs:Var_store.t -> config:Config.t -> t
end

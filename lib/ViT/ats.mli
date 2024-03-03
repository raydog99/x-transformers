open Torch

type t

val create : int -> eps:float -> t

val forward : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t
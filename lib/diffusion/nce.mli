open Torch

module type Model = sig
  val forward : Tensor.t -> Tensor.t
  val parameters : Tensor.t list
end

module NCE (M : Model) : sig
  val train : Model.t -> Tensor.t -> int -> int -> float -> unit
end

module GaussianModel : Model
open Torch

module type Opt = sig
  val n_epochs : int
  val batch_size : int
  val lr : float
  val b1 : float
  val b2 : float
  val n_cpu : int
  val latent_dim : int
  val n_classes : int
  val img_size : int
  val channels : int
  val sample_interval : int
end

module Make (O : Opt) : sig
  val img_shape : int * int * int

  module Generator : sig
    type t
    val create : unit -> t
    val forward : t -> Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t
  end

  module Discriminator : sig
    type t
    val create : unit -> t
    val forward : t -> Torch.Tensor.t -> Torch.Tensor.t -> Torch.Tensor.t
  end

  val train : unit -> unit
  val sample_image : int -> int -> unit
end
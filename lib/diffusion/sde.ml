open Torch

module type SDE = sig
  type t

  val create : int -> t
  val t : t -> float
  val sde : t -> Tensor.t -> float -> Tensor.t * Tensor.t
  val marginal_prob : t -> Tensor.t -> float -> Tensor.t * Tensor.t
  val prior_sampling : t -> Tensor.t -> Tensor.t
  val prior_logp : t -> Tensor.t -> Tensor.t
  val discretize : t -> Tensor.t -> float -> Tensor.t * Tensor.t
end

module MakeSDE (T : sig val n : int end) : SDE = struct
  type t = { n : int }

  let create n = { n }
  let t sde = float sde.n
  let sde sde x t =
    let beta_t = 0.1 +. t *. (20. -. 0.1) in
    let drift = Tensor.mul (Tensor.neg x) (Tensor.of_float (0.5 *. beta_t)) in
    let diffusion = Tensor.sqrt (Tensor.of_float beta_t) in
    drift, diffusion

  let marginal_prob sde x t =
    let log_mean_coeff = -0.25 *. t ** 2. *. (20. -. 0.1) -. 0.5 *. t *. 0.1 in
    let mean = Tensor.mul (Tensor.exp (Tensor.of_float log_mean_coeff)) x in
    let std = Tensor.sqrt (Tensor.sub (Tensor.of_float 1.) (Tensor.exp (Tensor.of_float (2. *. log_mean_coeff)))) in
    mean, std

  let prior_sampling _ rng_shape = Tensor.randn rng_shape
  let prior_logp _ z = Tensor.mul (Tensor.neg (Tensor.log (Tensor.mul (Tensor.of_float (2. *. Float.pi)) z))) (Tensor.of_float 0.5)

  let discretize sde x t =
    let timestep = int_of_float (t *. float (T.n - 1)) in
    let beta = Tensor.linspace (Tensor.of_float (0.1 /. float T.n)) (Tensor.of_float (20. /. float T.n)) (Tensor.of_int T.n) in
    let alpha = Tensor.sub (Tensor.of_float 1.) beta in
    let sqrt_beta = Tensor.sqrt beta in
    let f = Tensor.sub (Tensor.mul (Tensor.sqrt alpha) x) x in
    let g = sqrt_beta in
    f, g
end

module VPSDE = MakeSDE (struct let n = 1000 end)
module SubVPSDE = MakeSDE (struct let n = 1000 end)
open Torch

module TrainingConfig : sig
  type t

  val default_config : t
end
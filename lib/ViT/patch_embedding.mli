open Torch

module PatchEmbedding : sig
  type t

  val create :
    vs:Var_store.t ->
    image_size:int * int ->
    patch_size:int * int ->
    channels:int ->
    dim:int ->
    t

end
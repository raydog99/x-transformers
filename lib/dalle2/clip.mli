open Core
open Torch

val available_models : unit -> string list

val download : string -> string -> unit

val convert_image_to_rgb : 'a -> unit

val transform : int -> unit

val load :
  string ->
  ?device:string ->
  ?jit:bool ->
  ?download_root:string ->
  unit ->
  Torch.nn.Module * ('a -> torch.Tensor)
open Torch

let exists (v : 'a option) : bool = Option.is_some v

let default (v : 'a option) (d : 'a) : 'a = match v with
  | Some value -> value
  | None -> d

let l1norm ?(dim = -1) (t : Tensor.t) : Tensor.t = Tensor.normalize t ~p:1 ~dim

let is_distributed () : bool =
  if Dist.is_initialized () then
    Dist.get_world_size () > 1
  else
    false

let maybe_distributed_mean (t : Tensor.t) : Tensor.t =
  if not (is_distributed ()) then
    t
  else begin
    Dist.all_reduce t;
    t / Tensor.of_float (float_of_int (Dist.get_world_size ()))
  end
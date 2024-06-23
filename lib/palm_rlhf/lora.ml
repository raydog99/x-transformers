open Torch

let exists (v : 'a option) : bool =
  match v with
  | Some _ -> true
  | None -> false

let default (v : 'a option) (d : 'a) : 'a =
  match v with
  | Some x -> x
  | None -> d

class lora (dim : int) (dim_out : int) ?(r : int = 8) ?(alpha : int option) () =
  let alpha = default alpha r in
  object (self)
    val scale = float_of_int alpha /. float_of_int r
    val a = Tensor.randn [dim; r]
    val b = Tensor.zeros [r; dim_out]

    method weight = Tensor.(a @ b) * scale

    method forward (x : Tensor.t) : Tensor.t = Tensor.matmul x self#weight
  end

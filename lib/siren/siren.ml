open Torch

let exists v = Torch.(v <> none)

let cast_tuple ?(repeat=1) v =
  match v with
  | Tensor t -> t
  | _ -> Tensor.repeat v ~repeats:[|repeat|]

module Sine = struct
  let create w0 =
    fun x -> Tensor.sin (w0 * x)
end

module Siren = struct
  let create
      dim_in
      dim_out
      ~w0
      ~c
      ?(is_first=false)
      ?(use_bias=true)
      ?activation
      ?(dropout=0.)
    =
    let weight = Var (torch_zeros [dim_out; dim_in]) in
    let bias =
      if use_bias then Some (Var (torch_zeros [dim_out])) else None
    in
    let init_ weight bias c w0 =
      let dim = dim_in in
      let w_std =
        if is_first then
          1. /. float_of_int dim
        else
          sqrt (c /. float_of_int dim) /. w0
      in
      Tensor.uniform_ weight ~low:(-. w_std) ~high:w_std;
      match bias with
      | Some b -> Tensor.uniform_ b ~low:(-. w_std) ~high:w_std
      | None -> ()
    in
    fun x ->
      init_ weight bias c w0;
      let out = Layer.linear x weight bias in
      let activation_fn =
        match activation with
        | Some act -> act
        | None -> Sine.create w0
      in
      let out = activation_fn out in
      Layer.dropout out ~p:dropout
end
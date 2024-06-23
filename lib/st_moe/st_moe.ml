open Torch

(* Cumulative sum exclusive *)
let cumsum_exclusive t ~dim =
  assert (dim < 0);
  let num_pad_dims = -dim - 1 in
  let pre_padding = List.init num_pad_dims (fun _ -> 0, 0) |> List.concat in
  Tensor.(pad t ~pad:(List.append pre_padding [1, -1]) |> cumsum ~dim)

(* Log function with clamping *)
let log t ~eps =
  Tensor.(log (clamp_min t ~min:eps))

(* Gumbel noise *)
let gumbel_noise t =
  let noise = Tensor.zeros_like t |> Tensor.uniform_ ~from:0. ~to_:1. in
  Tensor.(- (log (- (log noise))))

(* Safe one hot encoding *)
let safe_one_hot indexes ~max_length =
  let max_index = Tensor.(indexes |> max |> to_int0_exn) + 1 in
  let one_hot_classes = max (max_index + 1) max_length in
  Tensor.(one_hot indexes ~num_classes:one_hot_classes |> narrow ~dim:(-1) ~start:0 ~length:max_length)

(* RMS normalization *)
module RMSNorm = struct
  type t = {
    scale : float;
    gamma : Tensor.t;
  }

  let create dim =
    let scale = Float.sqrt (Float.of_int dim) in
    let gamma = Tensor.ones [dim] |> Tensor.create in
    { scale; gamma }

  let forward t x =
    Tensor.(normalize x ~dim:(-1) * t.gamma * scalar t.scale)
end

(* GEGLU module *)
module GEGLU = struct
  type t = {
    mult_bias : Tensor.t;
  }

  let create ~dim ~mult_bias =
    let mult_bias = if mult_bias then Tensor.ones [dim] |> Tensor.create else Tensor.scalar 1. in
    { mult_bias }

  let forward t x =
    let x, gate = Tensor.chunk x 2 ~dim:(-1) in
    Tensor.(gelu gate * x * t.mult_bias)
end

(* Expert module *)
module Expert = struct
  type t = {
    net : Layer.t;
  }

  let create ~dim ~hidden_mult ~mult_bias ~prenorm =
    let dim_hidden = Int.of_float (Float.of_int dim *. hidden_mult *. 2. /. 3.) in
    let net =
      Layer.Sequential.create
        [
          (if prenorm then Some (Layer.of_fn (RMSNorm.forward (RMSNorm.create dim))) else None);
          Some (Layer.linear ~input_dim:dim ~output_dim:(dim_hidden * 2));
          Some (Layer.of_fn (GEGLU.forward (GEGLU.create ~dim:dim_hidden ~mult_bias)));
          Some (Layer.linear ~input_dim:dim_hidden ~output_dim:dim);
        ]
      |> List.filter_map Fun.id
    in
    { net }

  let init_ module_ =
    let init_linear module_ =
      let dim = Tensor.shape (Layer.Linear.weight module_) |> List.hd in
      let std = Float.(dim ** -0.5) in
      Layer.Linear.(weight module_ |> Tensor.uniform_ ~from:(-std) ~to:std);
      Layer.Linear.(bias module_ |> Tensor.uniform_ ~from:(-std) ~to:std)
    in
    match module_ with
    | Layer.Linear layer -> init_linear layer
    | _ -> ()

  let forward t x =
    Layer.forward t.net x
end
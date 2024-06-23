open Torch

let exists val_ = not (Tensor.equal val_ (Tensor.of_float 0.))

let default val_ d = if exists val_ then val_ else d

let rezero fn =
  let g = Tensor.zeros [1] |> Tensor.to_device (Tensor.device fn) |> nn.Parameter.of_tensor in
  fun x -> Tensor.mul (fn x) g

let scalenorm dim eps =
  let g = Tensor.ones [1] |> Tensor.to_device (Device.device_of_string "cpu") |> nn.Parameter.of_tensor in
  fun x ->
    let n = Tensor.norm x ~dim:(-1) ~keepdim:true |> Tensor.clamp_min eps in
    Tensor.div (Tensor.mul x (Tensor.reciprocal n)) g

let prenorm norm_class dim fn =
  let norm = norm_class dim in
  fun x ->
    let normalized = norm x in
    fn normalized

let chunk chunks fn along_dim =
  fun x ->
    if chunks = 1 then
      fn x
    else
      let chunked = Tensor.chunk x chunks ~dim:along_dim in
      Tensor.cat (List.map (fun c -> fn c) chunked) ~dim:along_dim

let rezero_module = rezero

let scalenorm_module dim eps = scalenorm dim eps

let prenorm_module norm_class dim fn = prenorm norm_class dim fn

let chunk_module chunks fn along_dim = chunk chunks fn along_dim
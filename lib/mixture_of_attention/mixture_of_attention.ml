open Torch

let exists v = Torch.(v <> none)

let default v d = if exists v then v else d

let pack_one t pattern = pack [ t ] pattern

let unpack_one t ps pattern = unpack t ps pattern |> fun tensors -> List.hd tensors

let pad_to_multiple tensor multiple ?(dim = -1) ?(value = 0.) () =
  let seq_len = Tensor.shape tensor |> fun shape -> List.nth shape dim in
  let m = float_of_int seq_len /. float_of_int multiple in
  if Float.is_integer m then (tensor, seq_len)
  else
    let remainder = ceil m *. float_of_int multiple -. float_of_int seq_len in
    let pad_offset = List.init (-1 - dim) (fun _ -> 0) @ [ 0; int_of_float remainder ] in
    let padded_tensor = F.pad tensor ~padding:(List.map ~f:float_of_int pad_offset) ~value () in
    (padded_tensor, seq_len)

let rmsNorm dim ?(groups = 1) () =
  let scale = Float.sqrt (float_of_int dim) in
  let gamma = nn_parameter (torch_ones [ groups; dim; 1 ]) in
  let forward x =
    let normed = F.normalize x ~dim:(-2) in
    normed * scale * gamma
  in
  forward
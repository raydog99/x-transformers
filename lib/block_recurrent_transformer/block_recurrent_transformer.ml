open Torch

let exists v = Torch.(v <> none)

let default v d = if exists v then v else d

let is_empty t = Tensor.numel t = 0

let cast_tuple t ~length = if Torch.is_tuple t then t else Tensor.repeat t length

let all_unique arr = List.length arr = List.length (List.dedup_and_sort ~compare:compare arr)

let eval_decorator fn =
  let inner self args kwargs =
    let was_training = self#is_training in
    self#eval ();
    let out = fn self args kwargs in
    self#train was_training;
    out
  in
  inner

let once fn =
  let called = ref false in
  fun x ->
    if !called then None
    else (
      called := true;
      Some (fn x)
    )

let print_once = once print

let compact arr = List.filter exists arr

let rec and_reduce = function
  | [] -> None
  | [ hd ] -> Some hd
  | hd :: tl -> Some (List.fold_left Tensor.( & ) hd tl)

let safe_cat ?(dim = 1) args =
  let args = compact args in
  match args with
  | [] -> None
  | _ -> Some (Tensor.cat args dim)

let divisible_by numer denom = numer mod denom = 0

let l2norm t = F.normalize t ~dim:(-1)

let pack_one t pattern = pack [ t ] pattern

let unpack_one t ps pattern = List.hd (unpack t ps pattern)

let pad_at_dim t pad ~dim ?(value = 0.) () =
  let dims_from_right = if dim < 0 then -dim - 1 else t#ndim - dim - 1 in
  let zeros = List.init dims_from_right (fun _ -> (0, 0)) |> List.flatten in
  F.pad t ~padding:(zeros @ pad) ~value

let layerNorm dim =
  let gamma = nn_parameter (torch_ones [ dim ]) in
  let beta = torch_zeros [ dim ] |> nn_buffer in
  let forward x = F.layer_norm x ~normalized_shape:(List.last_exn (Tensor.shape x)) ~weight:gamma ~bias:beta in
  forward
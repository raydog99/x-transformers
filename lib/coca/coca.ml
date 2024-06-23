open Base
open Torch

let pad_dim_to (t : Tensor.t) (length : int) ~(dim : int) : Tensor.t =
  let pad_length = length - Tensor.shape t.(dim) in
  let padding = List.init (Tensor.ndim t) ~f:(fun i -> if i = dim then (0, pad_length) else (0, 0)) in
  Tensor.pad t ~padding

let all_gather_variable_batch (t : Tensor.t) : Tensor.t * int list =
  let device = Tensor.device t in
  let rank = Dist.get_rank () in
  let world_size = Dist.get_world_size () in

  let size = Tensor.shape t.(0) |> Tensor.to_device ~device |> Tensor.to_type Int64 in
  let sizes = List.init world_size ~f:(fun _ -> Tensor.empty_like size) in
  let () = Dist.all_gather sizes size in

  let sizes = Tensor.stack sizes |> Tensor.to_list |> List.map ~f:Tensor.to_int_exn in
  let max_size = List.max_elt sizes ~compare:Int.compare |> Option.value_exn in

  let padded_t = pad_dim_to t max_size ~dim:0 in
  let gathered_tensors = List.init world_size ~f:(fun _ -> Tensor.empty_like padded_t) in
  let () = Dist.all_gather gathered_tensors padded_t in

  let gathered_tensor = Tensor.cat gathered_tensors ~dim:0 in
  let seq = Tensor.arange' 0 `L max_size ~device in

  let mask = Tensor.(seq < Tensor.unsqueeze (Tensor.of_list sizes) ~dim:1) in
  let mask = Tensor.reshape mask [| -1 |] in

  let gathered_tensor = Tensor.boolean_mask gathered_tensor mask in

  gathered_tensor, sizes

module AllGather : sig
  val forward : Tensor.t -> Tensor.t
  val backward : Tensor.t -> Tensor.t
end = struct
  let forward (x : Tensor.t) : Tensor.t =
    assert (Dist.is_initialized () && Dist.get_world_size () > 1);
    let x, batch_sizes = all_gather_variable_batch x in
    x

  let backward (grads : Tensor.t) : Tensor.t =
    let batch_sizes = Dist.get_rank () in
    Tensor.split_with_sizes grads ~split_sizes:(List.map ~f:Long.of_int batch_sizes) ~dim:0
    |> fun grads_by_rank -> List.nth_exn grads_by_rank (Dist.get_rank ())
end
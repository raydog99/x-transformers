module type Module = sig
  type t

  val forward : t -> Tensor.t -> Tensor.t -> Tensor.t option -> bool -> bool -> Tensor.t * (Tensor.t -> Tensor.t)
end

module FrameAverage (Module : Module) = struct
  type t = {
    net : Module.t option;
    dim : int;
    num_frames : int;
    operations : Tensor.t;
    stochastic : bool;
    return_stochastic_as_augmented_pos : bool;
    invariant_output : bool;
  }

  let init
      ?(net : Module.t option)
      ~dim
      ?(stochastic : bool = false)
      ?(invariant_output : bool = false)
      ?(return_stochastic_as_augmented_pos : bool = false)
      () =
    assert (dim > 1);

    let num_frames = int_of_float (2. ** float dim) in

    let directions = Tensor.of_array [|-1.; 1.|] in
    let accum = ref [] in

    for ind = 0 to dim - 1 do
      let dim_slice = Array.make dim None in
      dim_slice.(ind) <- Some (Tensor.colon);
      let accum_tensor = Tensor.broadcast_tensors directions dim_slice in
      accum := accum_tensor :: !accum
    done;

    let operations = Tensor.stack (List.rev !accum) ~dim:(-1) in
    assert (Tensor.shape operations = [|num_frames; dim|]);

    {
      net;
      dim;
      num_frames;
      operations;
      stochastic;
      return_stochastic_as_augmented_pos;
      invariant_output;
    }

  let forward self points ?(frame_average_mask : Tensor.t option) ?(return_framed_inputs_and_averaging_function : bool = false) () =
    assert (Tensor.shape points.(|-1|) = self.dim);

    let frame_average_mask =
      match frame_average_mask with
      | Some mask -> Tensor.rearrange mask ~pattern:"... -> ... 1" |> Tensor.mul points
      | None -> points
    in

    let batch, seq_dim, input_dim = Tensor.shape points in

    let num, den, centroid =
      match frame_average_mask with
      | Some mask ->
          let num = Tensor.reduce points ~pattern:"b n d -> b 1 d" `Sum in
          let den = Tensor.reduce mask ~pattern:"b n 1 -> b 1 1" `Sum |> Tensor.clamp_min 1. in
          let centroid = Tensor.div num den in
          (num, den, centroid)
      | None ->
          let centroid = Tensor.reduce points ~pattern:"b n d -> b 1 d" `Mean in
          (Tensor.zero, Tensor.zero, centroid)
    in

    let centered_points = Tensor.sub points centroid in

    let centered_points =
      match frame_average_mask with
      | Some mask -> Tensor.mul centered_points mask
      | None -> centered_points
    in

    let covariance = Tensor.einsum centered_points centered_points ~pattern:"b n d, b n e -> b d e" in
    let _, eigenvectors = Tensor.linalg_eigh covariance in

    let num_frames = if self.stochastic then 1 else self.num_frames in
    let operations = if self.stochastic then Tensor.get_slice self.operations [| [|(Random.int self.num_frames)|]; [||] |] else self.operations in

    let frames = Tensor.mul (Tensor.rearrange eigenvectors ~pattern:"b d e -> b 1 d e") (Tensor.rearrange operations ~pattern:"f e -> f 1 e") in

    let inputs = Tensor.einsum frames centered_points ~pattern:"b f d e, b n d -> b f n e" in

    let frame_average out =
      let out =
        if not self.invariant_output then
          Tensor.einsum frames out ~pattern:"b f d e, b f ... e -> b f ... d"
        else
          out
      in
      if not self.stochastic then
        Tensor.reduce out ~pattern:"b f ... -> b ..." `Mean
      else
        Tensor.rearrange out ~pattern:"b 1 ..." ~dims:[|batch; seq_dim; input_dim|]
    in

    if return_framed_inputs_and_averaging_function || Option.is_none self.net then
      if self.stochastic && self.return_stochastic_as_augmented_pos then
        Tensor.rearrange inputs ~pattern:"b 1 ..." ~dims:[|batch; seq_dim; input_dim|]
      else
        inputs, frame_average

    else
      let inputs = Tensor.rearrange inputs ~pattern:"b f ... -> (b f) ..." ~dims:[|batch * num_frames; seq_dim; input_dim|] in

      let out = Module.forward self.net inputs () in

      let out =
        Tensor.tree_map
          (fun t ->
             if Tensor.is_tensor t then
               Tensor.rearrange t ~pattern:"(b f) ..." ~dims:[|batch; num_frames; seq_dim; input_dim|]
             else
               t)
          out
      in
      Tensor.tree_map (fun t -> if Tensor.is_tensor t then frame_average t else t) out
end
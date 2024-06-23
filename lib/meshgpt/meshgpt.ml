open Torch

module ContinuousEmbed = struct
  type t = Layer.t

  let create dim_cont = 
    let module Rearrange = struct
      let forward x = Tensor.unsqueeze x ~dim:1
    end in
    Layer.of_list [
      Layer.fn Rearrange.forward;
      Layer.linear ~input_dim:1 dim_cont;
      Layer.silu ();
      Layer.linear ~input_dim:dim_cont dim_cont;
      Layer.layer_norm dim_cont
    ]

  let forward model x = Layer.forward model x
end

module DeriveAngle = struct
  let l2norm x = 
    let norm = Tensor.(sqrt (sum (pow x 2.0) ~dim:[-1] ~keepdim:true)) in
    Tensor.(x / norm)

  let forward x y ~eps =
    let z = Tensor.einsum2 ~equation:"... d, ... d -> ..." (l2norm x) (l2norm y) in
    Tensor.(z |> clip ~min:(-1.0 +. eps) ~max:(1.0 -. eps) |> acos)
end

module GetDerivedFaceFeatures = struct
  let l2norm x = 
    let norm = Tensor.(sqrt (sum (pow x 2.0) ~dim:[-1] ~keepdim:true)) in
    Tensor.(x / norm)

  let forward face_coords =
    let shifted_face_coords = Tensor.cat [Tensor.narrow face_coords ~dim:2 ~start:(Tensor.size face_coords 2 - 1) ~length:1;
                                          Tensor.narrow face_coords ~dim:2 ~start:0 ~length:(Tensor.size face_coords 2 - 1)] ~dim:2 in

    let angles = DeriveAngle.forward face_coords shifted_face_coords ~eps:1e-5 in

    let edges = Tensor.unbind (Tensor.(face_coords - shifted_face_coords)) ~dim:2 in
    let edge1 = List.nth edges 0 in
    let edge2 = List.nth edges 1 in
    let cross_product = Tensor.cross edge1 edge2 ~dim:(-1) in

    let normals = l2norm cross_product in
    let area = Tensor.(norm cross_product ~dim:[-1] ~keepdim:true * f 0.5) in

    `Assoc [
      "angles", angles;
      "area", area;
      "normals", normals
    ]
end

module Discretize = struct
  let forward t ~continuous_range ~num_discrete =
    let lo, hi = continuous_range in
    assert (hi > lo);

    let t = Tensor.(t - f lo) / Tensor.(f (hi -. lo)) in
    let t = Tensor.(t * f (float_of_int num_discrete)) in
    let t = Tensor.(t - f 0.5) in

    Tensor.(round t |> to_type ~type_:TInt64 |> clamp ~min:0 ~max:(num_discrete - 1))
end
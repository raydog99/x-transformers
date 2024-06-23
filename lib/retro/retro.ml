open Base
open Torch

let deepnorm_init transformer beta module_name_match_list =
  Module.named_modules transformer
  |> List.iter ~f:(fun (name, module') ->
         match module' with
         | Linear module' ->
             let needs_beta_gain =
               List.exists module_name_match_list ~f:(fun substr -> String.is_substring name ~substring:substr)
             in
             let gain = if needs_beta_gain then beta else 1. in
             nn_init_xavier_normal_ module' ~gain;
             (match module'.bias with
             | Some bias -> nn_init_constant_ bias ~value:0.
             | None -> ())
         | _ -> ())

class rmsNorm dim ~eps ~gated =
  object
    val eps = eps
    val scale = Float.pow (Float.of_int dim) (-0.5)

    val gamma = 
      let gamma = Tensor.ones [|dim|] in
      nn.Parameter.create gamma

    val weight = 
      match gated with
      | true -> 
        let weight = Tensor.ones [|dim|] in
        nn.Parameter.create weight
      | false -> None

    method forward x =
      let norm = Tensor.norm x ~keepdim:true ~dim:(-1) |> Tensor.mul_scalar scale in
      let clamped_norm = Tensor.clamp_min norm eps in
      let out = Tensor.div x clamped_norm |> Tensor.mul gamma in
      match weight with
      | Some w -> Tensor.mul out (Tensor.sigmoid (Tensor.mul x w))
      | None -> out
  end
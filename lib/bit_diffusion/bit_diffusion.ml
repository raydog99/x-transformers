open Base
open Torch

let bits = 8

let exists x = Option.is_some x

let default val d =
  match val with
  | Some v -> v
  | None -> (match d () with | Some v -> v | None -> d)

let cycle dl = Sequence.cycle dl

let has_int_squareroot num = Float.(equal (sqrt num |> square) num)

let num_to_groups num divisor =
  let groups = num / divisor in
  let remainder = num % divisor in
  let arr = List.init groups ~f:(fun _ -> divisor) in
  if remainder > 0 then arr @ [remainder] else arr

let convert_image_to pil_img_type image =
  if Image.mode image <> pil_img_type then Image.to_image image pil_img_type else image

let residual fn x args kwargs = fn x args kwargs + x

let upsample dim dim_out =
  sequential
    [
      upsample_nearest2d ~scale_factor:2;
      conv2d ~padding:1 ~stride:1 dim (default dim_out dim) ~kernel_size:3;
    ]

let downsample dim dim_out =
  conv2d ~padding:1 ~stride:2 dim (default dim_out dim) ~kernel_size:4
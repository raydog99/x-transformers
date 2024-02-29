open Base
open Torch
open TwinsSVT

(* Dataset: non-small cell lung cancer histopathology, ISICDM '21 *)
let evaluate_twins_model ~image_set_basename ~nbr_of_classes ~labels_names =
  let data_dir = "/3.testing/img" in
  let input_path = Filename.concat data_dir (image_set_basename ^ "*") in
  let data_files = Sys.readdir input_path in

  let count_slides = ref 0 in

  let label_names =
    In_channel.read_lines labels_names
    |> List.filter ~f:(fun line -> not (String.is_empty line))
  in

  for i = 0 to Array.length data_files - 1 do
    let next_slide = data_files.(i) in
    printf "New Slide ------------ %d\n" !count_slides;

    let label_index = int_of_string (List.last_exn (String.split ~on:'_' next_slide) |> String.split ~on:'.' |> List.hd_exn) in
    let label_name = List.nth_exn label_names label_index in

    printf "label %d: %s\n" label_index label_name;

    let dummy_input = Tensor.randn [||] in

    let model_config = ModelConfig.create ~num_classes:nbr_of_classes in
    let training_config = TrainingConfig.create () in
    let twins_model = TwinsSVT.create ~num_classes:nbr_of_classes in

    let output = TwinsSVT.forward twins_model dummy_input in

    count_slides := !count_slides + 1;
  done

let () =
  let image_set_basename = "test_" in
  let nbr_of_classes = 10 in
  let labels_names = "/Lung/Test_All512pxTiled/9_10mutations/label_names.txt" in
  evaluate_twins_model ~image_set_basename ~nbr_of_classes ~labels_names
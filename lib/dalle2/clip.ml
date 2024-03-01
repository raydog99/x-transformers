open Core
open Torch

let models = [
    "RN50", "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt";
    "RN101", "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt";
    "RN50x4", "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt";
    "RN50x16", "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt";
    "RN50x64", "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt";
    "ViT-B/32", "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt";
    "ViT-B/16", "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt";
    "ViT-L/14", "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt";
    "ViT-L/14@336px", "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt";
]

let available_models () = []

let download url root =
  Printf.printf "Downloading model from %s to %s\n" url root;

let convert_image_to_rgb image =
  Printf.printf "Converting image to RGB\n";

let transform n_px =
  Printf.printf "Transforming with %d pixels\n" n_px;

let load name ?(device="cuda") ?(jit=false) ?download_root () =
  match List.assoc name models with
  | exception Not_found -> failwith (Printf.sprintf "Model %s not found; available models = %s" name (String.concat ~sep:", " (available_models ())))
  | model_url ->
    let model_path = download model_url (Option.value ~default:"~/.cache/clip" download_root) in
    let opened_file = Stdlib.open_in_bin model_path in
    try
      let model, state_dict =
        try
          Torch.jit_load opened_file ~map_location:(if jit then device else "cpu") |> Torch.jit_to torch_nn_module |> torch_nn_module_eval, None
        with
        | _ ->
          if jit then Printf.eprintf "File %s is not a JIT archive. Loading as a state dict instead\n" model_path;
          torch_load opened_file ~map_location:"cpu", opened_file in
      end
    with
    | e -> In_channel.close opened_file; raise e
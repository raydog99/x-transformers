open Torch

let open_memmap filename dtype =
  Torch.Tensor.load_npy filename

module ReplayMemoryDataset = struct
  type t = {
    text_embeds: Tensor.t;
    states: Tensor.t;
    actions: Tensor.t;
    rewards: Tensor.t;
    dones: Tensor.t;
    episode_length: int array;
    indices: (int * int) array;
    num_timesteps: int;
    num_episodes: int;
    max_episode_len: int;
  }

  let create ?(folder="default_replay_memories_folder") num_timesteps =
    assert (num_timesteps >= 1);
    
    let is_single_timestep = num_timesteps = 1 in
    
    let text_embeds_path = Filename.concat folder text_embeds_filename in
    let states_path = Filename.concat folder states_filename in
    let actions_path = Filename.concat folder actions_filename in
    let rewards_path = Filename.concat folder rewards_filename in
    let dones_path = Filename.concat folder dones_filename in

    let text_embeds = open_memmap text_embeds_path Float32 in
    let states = open_memmap states_path Float32 in
    let actions = open_memmap actions_path Int32 in
    let rewards = open_memmap rewards_path Float32 in
    let dones = open_memmap dones_path Bool in

    let dones_cumsum = Tensor.cumsum ~dim:(-1) dones in
    let episode_length = Tensor.sum1 ~dim:[-1] (Tensor.eq_scalar dones_cumsum 0.) ~dtype:(T Int64) |> Tensor.to_int1_exn in
    Array.iteri (fun i v -> episode_length.(i) <- v + 1) episode_length;
    
    let trainable_episode_indices = Array.mapi (fun i v -> if v >= num_timesteps then true else false) episode_length in
    
    let filter_indices indices =
      Array.fold_left (fun acc (i, v) -> if trainable_episode_indices.(i) then v :: acc else acc) [] (Array.mapi (fun i v -> (i, v)) indices) |> Array.of_list
    in

    let text_embeds = Tensor.index_select text_embeds ~dim:0 (Tensor.of_int_list (filter_indices text_embeds)) in
    let states = Tensor.index_select states ~dim:0 (Tensor.of_int_list (filter_indices states)) in
    let actions = Tensor.index_select actions ~dim:0 (Tensor.of_int_list (filter_indices actions)) in
    let rewards = Tensor.index_select rewards ~dim:0 (Tensor.of_int_list (filter_indices rewards)) in
    let dones = Tensor.index_select dones ~dim:0 (Tensor.of_int_list (filter_indices dones)) in
    let episode_length = filter_indices episode_length in
    
    assert (Array.length dones > 0);
    
    let num_episodes = Array.length dones in
    let max_episode_len = Array.fold_left max 0 episode_length in

    let timestep_arange = Tensor.arange ~end_:(Scalar.float_of_int (max_episode_len - 1)) ~options:(T Float32, Cpu) |> Tensor.to_int1_exn in

    let timestep_indices = 
      Array.init num_episodes (fun i -> 
        Array.init max_episode_len (fun j -> 
          (i, timestep_arange.(j))
        )
      ) |> Array.concat
    in

    let trainable_mask = 
      Array.init num_episodes (fun i ->
        Array.map (fun x -> x < (episode_length.(i) - num_timesteps)) timestep_arange
      ) |> Array.concat
    in

    let indices = Array.of_list (List.filteri (fun i _ -> trainable_mask.(i)) (Array.to_list timestep_indices)) in

    { text_embeds; states; actions; rewards; dones; episode_length; indices; num_timesteps; num_episodes; max_episode_len }

  let length t = Array.length t.indices

  let get_item t idx =
    let episode_index, timestep_index = t.indices.(idx) in
    let timestep_slice = (timestep_index, timestep_index + t.num_timesteps - 1) in

    let slice_tensor tensor (start_idx, end_idx) =
      Tensor.narrow tensor ~dim:1 ~start:start_idx ~length:(end_idx - start_idx + 1)
    in

    let text_embeds = slice_tensor t.text_embeds timestep_slice |> Tensor.copy in
    let states = slice_tensor t.states timestep_slice |> Tensor.copy in
    let actions = slice_tensor t.actions timestep_slice |> Tensor.copy in
    let rewards = slice_tensor t.rewards timestep_slice |> Tensor.copy in
    let dones = slice_tensor t.dones timestep_slice |> Tensor.copy in

    let next_state = 
      if timestep_index < (t.max_episode_len - 1) then
        Tensor.index t.states [|episode_index; timestep_index + 1|] |> Tensor.copy
      else
        Tensor.index t.states [|episode_index; timestep_index|] |> Tensor.copy
    in

    text_embeds, states, actions, next_state, rewards, dones
end
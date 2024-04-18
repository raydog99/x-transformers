open Base

let routing_attention q k v num_clusters =
  let n = Array.length q in
  let d = Array.length q.(0) in
  let centroids = Array.make num_clusters (Array.make d 0.) in

  (* Normalize queries and keys *)
  let q = Array.map (fun q_vec ->
    let norm = sqrt (Array.fold_left (+.) 0. (Array.map (fun x -> x ** 2.) q_vec)) in
    Array.map (fun x -> x /. norm) q_vec
  ) q in
  let k = Array.map (fun k_vec ->
    let norm = sqrt (Array.fold_left (+.) 0. (Array.map (fun x -> x ** 2.) k_vec)) in
    Array.map (fun x -> x /. norm) k_vec
  ) k in

  (* Compute distances to centroids *)
  let q_dists = Array.make_matrix n num_clusters max_float in
  let k_dists = Array.make_matrix n num_clusters max_float in
  for i = 0 to n - 1 do
    for j = 0 to num_clusters - 1 do
      q_dists.(i).(j) <- Array.fold_left2 (fun acc x y -> acc +. (x -. y) ** 2.) 0. q.(i) centroids.(j);
      k_dists.(i).(j) <- Array.fold_left2 (fun acc x y -> acc +. (x -. y) ** 2.) 0. k.(i) centroids.(j);
    done
  done;

  (* Assign queries and keys to clusters *)
  let q_clusters = Array.mapi (fun i q_vec -> Array.sort (fun (_, d1) (_, d2) -> compare d1 d2) (Array.mapi (fun j d -> (j, d)) q_dists.(i)) |> Array.to_list |> List.take (n / num_clusters) |> List.map fst) q in
  let k_clusters = Array.mapi (fun i k_vec -> Array.sort (fun (_, d1) (_, d2) -> compare d1 d2) (Array.mapi (fun j d -> (j, d)) k_dists.(i)) |> Array.to_list |> List.take (n / num_clusters) |> List.map fst) k in

  (* Compute attention scores *)
  let scores = Array.make_matrix num_clusters (n / num_clusters) 0. in
  for i = 0 to num_clusters - 1 do
    for j = 0 to n / num_clusters - 1 do
      let q_idx = q_clusters.(i).(j) in
      let k_idx = k_clusters.(i).(j) in
      scores.(i).(j) <- Array.fold_left2 (+.) 0. q.(q_idx) k.(k_idx)
    done
  done;

  (* Apply softmax to get attention weights *)
  let weights = Array.map (fun row ->
    let row_sum = Array.fold_left (+.) 0. (Array.map exp row) in
    Array.map (fun x -> exp x /. row_sum) row
  ) scores in

  (* Compute weighted sum of values *)
  let output = Array.make n (Array.make d 0.) in
  for i = 0 to num_clusters - 1 do
    for j = 0 to n / num_clusters - 1 do
      let q_idx = q_clusters.(i).(j) in
      let k_idx = k_clusters.(i).(j) in
      for d_idx = 0 to d - 1 do
        output.(q_idx).(d_idx) <- output.(q_idx).(d_idx) +. weights.(i).(j) *. v.(k_idx).(d_idx)
      done
    done
  done;

  output
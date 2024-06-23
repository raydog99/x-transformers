let get_rank () =
  if Distributed.is_initialized () then Distributed.get_rank () else 0

let get_world_size () =
  if Distributed.is_initialized () then Distributed.get_world_size () else 1

let is_distributed () =
  Distributed.is_initialized () && Distributed.get_world_size () > 1

let circular_index_left pos ring_size ?(num=1) () =
  ((pos - num) + ring_size) % ring_size

let circular_index_right pos ring_size ?(num=1) () =
  (pos + num) % ring_size

let circular_rank_left ?rank ?ring_size ?(num=1) () =
  let rank = match rank with Some r -> r | None -> get_rank () in
  let ring_size = match ring_size with Some rs -> rs | None -> get_world_size () in
  let ring_set_num = rank / ring_size in
  let offset = ring_set_num * ring_size in
  circular_index_left rank ring_size ~num () + offset

let circular_rank_right ?rank ?ring_size ?(num=1) () =
  let rank = match rank with Some r -> r | None -> get_rank () in
  let ring_size = match ring_size with Some rs -> rs | None -> get_world_size () in
  let ring_set_num = rank / ring_size in
  let offset = ring_set_num * ring_size in
  circular_index_right rank ring_size ~num () + offset
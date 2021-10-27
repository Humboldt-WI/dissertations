#### Metrics calculation for p-space and  undirected network for all companies

# Data preparation ----

# gathering edges of three companies and adding nodes attributes

edges <- rbind(edges_FLIX_pspace, edges_OUIB_pspace, edges_EURL_pspace) %>%
  dplyr::mutate(edge_id = row_number()) %>%
  left_join((nodes %>% dplyr::select(node_id, stop_id)), by=c("departure_placecode"="stop_id")) %>%
  dplyr::rename(from = node_id) %>%
  left_join((nodes %>% dplyr::select(node_id, stop_id)), by=c("arrival_placecode"="stop_id")) %>%
  dplyr::rename(to = node_id) %>%
  ungroup() %>%
  dplyr::select(-departure_placecode, -arrival_placecode)

# adding geo details to nodes and grouping edges together by arr-s

edges_arr <- edges %>%
  left_join((nodes %>% left_join(geo_data, by=c("insee_code"="com")) %>% dplyr::select(node_id, arr)), by=c("from"="node_id")) %>%
  dplyr::rename(arr_from = arr) %>%
  left_join((nodes %>% left_join(geo_data, by=c("insee_code"="com")) %>% dplyr::select(node_id, arr)), by=c("to"="node_id")) %>%
  dplyr::rename(arr_to = arr) %>%
  group_by(arr_from, arr_to) %>%
  dplyr::summarize(count_routes=n(), frequency=sum(frequency)) %>% 
  ungroup()  %>% 
  left_join(nodes_arr, by=c("arr_to"="arr")) %>%
  left_join(nodes_arr, by=c("arr_from"="arr"), suffix=c("_from", "_to")) %>% 
  mutate(arr_lat_from = as.double(arr_lat_from), arr_lon_from=as.double(arr_lon_from),
         arr_lat_to = as.double(arr_lat_to), arr_lon_to=as.double(arr_lon_to)) %>% 
  ungroup()

# calculating distance between arr-s
edges_arr$distance <- round(geosphere::distVincentyEllipsoid(edges_arr[,c("arr_lat_from","arr_lon_from")], edges_arr[,c("arr_lat_to","arr_lon_to")])/1000,2)
edges_arr$distance <- as.integer(edges_arr$distance)
edges_arr <- edges_arr %>% 
  dplyr::select(arr_from, arr_to, frequency, distance)

# returning undirected edges
tmp1 <- edges_arr %>%
  dplyr::select(arr_from, arr_to, frequency, distance) %>%
  filter(arr_from <= arr_to) %>% 
  dplyr::rename(x=1, y=2)

tmp2 <- edges_arr %>%
  dplyr::select(arr_to, arr_from, frequency, distance) %>%
  filter(arr_from > arr_to) %>% 
  dplyr::rename(x=1, y=2)

edges_arr_un <- bind_rows(tmp1, tmp2) %>% 
  dplyr::group_by(x,y) %>% 
  dplyr::summarise(frequency=min(frequency), distance=mean(distance)) 

# finding connected arr-s
nodes_arr_connected <- edges_arr %>% ungroup()%>% dplyr::select(arr_to) %>% dplyr::rename(arr=1) %>% 
  bind_rows(edges_arr %>% ungroup()%>% dplyr::select(arr_from) %>% dplyr::rename(arr=1)) %>%
  distinct(arr) %>%
  left_join(nodes_arr, by = "arr") %>%
  drop_na() %>% 
  dplyr::mutate(id = row_number())

# constructing set of all possible combinations of arr-s in undirected network
edges_arr_un_possible <- arrondissements %>% 
  mutate(arr_from=arr, arr_to=arr) %>% 
  expand(arr_from, arr_to) %>%
  filter(arr_from < arr_to) %>%
  left_join(arrondissements %>% dplyr::select(arr, arr_center), by=c("arr_from"="arr")) %>%
  left_join(arrondissements %>% dplyr::select(arr, arr_center), by=c("arr_to"="arr"), suffix=c("_from", "_to")) %>%
  left_join(arr, by=c("arr_from"="arr")) %>%
  left_join(arr, by=c("arr_to"="arr"), suffix=c("_from", "_to")) %>% 
  mutate(arr_lat_from = as.double(arr_lat_from), arr_lon_from=as.double(arr_lon_from),
         arr_lat_to = as.double(arr_lat_to), arr_lon_to=as.double(arr_lon_to))

edges_arr_un_possible$distance <- round(geosphere::distVincentyEllipsoid(edges_arr_un_possible[,c("arr_lat_from","arr_lon_from")], edges_arr_un_possible[,c("arr_lat_to","arr_lon_to")])/1000,2)
edges_arr_un_possible$distance <- as.integer(edges_arr_un_possible$distance)

edges_arr_un_possible <- edges_arr_un_possible %>%
  filter(distance>100) %>%
  left_join(edges_arr %>% dplyr::select(arr_to, arr_from, frequency), by=c("arr_to"="arr_to", "arr_from"="arr_from")) %>%
  mutate(frequency=ifelse(is.na(frequency)==TRUE,0, frequency),
         link=ifelse(frequency==0,0,1)) %>%
  dplyr::select(arr_from, arr_to, arr_total_pop_from, arr_total_pop_to, distance, frequency, link) %>%
  dplyr::mutate(id = row_number()) %>% 
  dplyr::rename(total_pop_to =  arr_total_pop_to, total_pop_from = arr_total_pop_from, from = arr_from, to = arr_to)

edges_df <- edges_arr_un_possible

# edges, nodes - by stations
# edges_arr, nodes_arr - by arr-s
# edges_arr_un, nodes_arr - by arr-s, undirected
# nodes_arr_connected - only those arr's which are connected in the studied network
# edges_arr_un_possible - all possible edges between arr-s



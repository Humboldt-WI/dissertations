#### Metrics calculation for l-space and  undirected network for all companies

# Data preparation ----

# gathering edges of three companies and adding nodes attributes

edges <- rbind(edges_FLIX_lspace, edges_OUIB_lspace, edges_EURL_lspace) %>%
  dplyr::mutate(edge_id = row_number()) %>%
  left_join((nodes %>% dplyr::select(node_id, stop_id)), by=c("departure_placecode"="stop_id")) %>%
  dplyr::rename(from = node_id) %>%
  left_join((nodes %>% dplyr::select(node_id, stop_id)), by=c("arrival_placecode"="stop_id")) %>%
  dplyr::rename(to = node_id) %>%
  ungroup() %>%
  dplyr::select(-departure_placecode, -arrival_placecode)

# adding geo details to nodes and grouping edges together by dep-s

edges_dep <- edges %>%
  left_join((nodes %>% left_join(geo_data, by=c("insee_code"="com")) %>% dplyr::select(node_id, dep)), by=c("from"="node_id")) %>%
  dplyr::rename(dep_from = dep) %>%
  left_join((nodes %>% left_join(geo_data, by=c("insee_code"="com")) %>% dplyr::select(node_id, dep)), by=c("to"="node_id")) %>%
  dplyr::rename(dep_to = dep) %>%
  group_by(dep_from, dep_to) %>%
  dplyr::summarize(count_routes=n(), frequency=sum(frequency)) %>% 
  ungroup()  %>% 
  left_join(nodes_dep, by=c("dep_to"="dep")) %>%
  left_join(nodes_dep, by=c("dep_from"="dep"), suffix=c("_from", "_to")) %>% 
  mutate(dep_lat_from = as.double(dep_lat_from), dep_lon_from=as.double(dep_lon_from),
         dep_lat_to = as.double(dep_lat_to), dep_lon_to=as.double(dep_lon_to)) %>% 
  ungroup()

# calculating distance between dep-s
edges_dep$distance <- round(geosphere::distVincentyEllipsoid(edges_dep[,c("dep_lat_from","dep_lon_from")], edges_dep[,c("dep_lat_to","dep_lon_to")])/1000,2)
edges_dep$distance <- as.integer(edges_dep$distance)
edges_dep <- edges_dep %>% 
  dplyr::select(dep_from, dep_to, frequency, distance)

# returning undirected edges
tmp1 <- edges_dep %>%
  dplyr::select(dep_from, dep_to, frequency, distance) %>%
  filter(dep_from <= dep_to) %>% 
  dplyr::rename(x=1, y=2)

tmp2 <- edges_dep %>%
  dplyr::select(dep_to, dep_from, frequency, distance) %>%
  filter(dep_from > dep_to) %>% 
  dplyr::rename(x=1, y=2)

edges_dep_un <- bind_rows(tmp1, tmp2) %>% 
  dplyr::group_by(x,y) %>% 
  dplyr::summarise(frequency=min(frequency), distance=mean(distance)) 

# finding connected dep-s
nodes_dep_connected <- edges_dep %>% ungroup()%>% dplyr::select(dep_to) %>% dplyr::rename(dep=1) %>% 
  bind_rows(edges_dep %>% ungroup()%>% dplyr::select(dep_from) %>% dplyr::rename(dep=1)) %>%
  distinct(dep) %>%
  left_join(nodes_dep, by = "dep") %>%
  drop_na() %>% 
  dplyr::mutate(id = row_number())

# constructing set of all possible combinations of dep-s in undirected network
edges_dep_un_possible <- departments %>% 
  mutate(dep_from=dep, dep_to=dep) %>% 
  expand(dep_from, dep_to) %>%
  filter(dep_from < dep_to) %>%
  left_join(departments %>% dplyr::select(dep, dep_center), by=c("dep_from"="dep")) %>%
  left_join(departments %>% dplyr::select(dep, dep_center), by=c("dep_to"="dep"), suffix=c("_from", "_to")) %>%
  left_join(dep, by=c("dep_from"="dep")) %>%
  left_join(dep, by=c("dep_to"="dep"), suffix=c("_from", "_to")) %>% 
  mutate(dep_lat_from = as.double(dep_lat_from), dep_lon_from=as.double(dep_lon_from),
         dep_lat_to = as.double(dep_lat_to), dep_lon_to=as.double(dep_lon_to))

edges_dep_un_possible$distance <- round(geosphere::distVincentyEllipsoid(edges_dep_un_possible[,c("dep_lat_from","dep_lon_from")], edges_dep_un_possible[,c("dep_lat_to","dep_lon_to")])/1000,2)
edges_dep_un_possible$distance <- as.integer(edges_dep_un_possible$distance)

edges_dep_un_possible <- edges_dep_un_possible %>%
  filter(distance>100) %>%
  left_join(edges_dep %>% dplyr::select(dep_to, dep_from, frequency), by=c("dep_to"="dep_to", "dep_from"="dep_from")) %>%
  mutate(frequency=ifelse(is.na(frequency)==TRUE,0, frequency),
         link=ifelse(frequency==0,0,1))  %>%
  dplyr::select(dep_from, dep_to, dep_total_pop_from, dep_total_pop_to, distance, frequency, link) %>%
  dplyr::mutate(id = row_number()) %>% 
  dplyr::rename(total_pop_to =  dep_total_pop_to, total_pop_from = dep_total_pop_from, from = dep_from, to = dep_to)

edges_df <- edges_dep_un_possible

# edges, nodes - by stations
# edges_dep, nodes_dep - by dep-s
# edges_dep_un, nodes_dep - by dep-s, undirected
# nodes_dep_connected - only those dep's which are connected in the studied network
# edges_dep_un_possible - all possible edges between dep-s





